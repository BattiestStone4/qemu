/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 这是一个教育用途的 GPGPU 设备模拟实现。
 * 参考了 Vortex GPGPU 的 SIMT 架构设计。
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"

/*
 * ============================================================================
 * 第一部分: 中断处理
 * ============================================================================
 * GPGPU 支持 MSI-X 中断，这是现代 PCIe 设备的标准中断机制。
 * 当设备完成操作（如 DMA 传输或内核执行）时，会触发相应的中断通知驱动程序。
 */

/**
 * gpgpu_update_irq - 更新设备中断状态
 * @s: GPGPU 设备状态
 *
 * 根据 irq_status 和 irq_enable 寄存器的值来决定是否触发中断。
 * 只有当某个中断位同时在 status 和 enable 中被设置时，才会产生中断。
 */
static void gpgpu_update_irq(GPGPUState *s)
{
    uint32_t pending = s->irq_status & s->irq_enable;

    if (pending) {
        /* 有挂起的中断，触发 MSI-X */
        if (msix_enabled(&s->parent_obj)) {
            /* 根据中断类型选择不同的向量 */
            if (pending & GPGPU_IRQ_KERNEL_DONE) {
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_KERNEL);
            }
            if (pending & GPGPU_IRQ_DMA_DONE) {
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_DMA);
            }
            if (pending & GPGPU_IRQ_ERROR) {
                msix_notify(&s->parent_obj, GPGPU_MSIX_VEC_ERROR);
            }
        } else if (msi_enabled(&s->parent_obj)) {
            /* 回退到 MSI */
            msi_notify(&s->parent_obj, 0);
        } else {
            /* 回退到传统 INTx 中断 */
            pci_set_irq(&s->parent_obj, 1);
        }
    } else {
        /* 没有挂起的中断，降低 INTx 电平 */
        if (!msix_enabled(&s->parent_obj) && !msi_enabled(&s->parent_obj)) {
            pci_set_irq(&s->parent_obj, 0);
        }
    }
}

/**
 * gpgpu_raise_irq - 触发指定中断
 * @s: GPGPU 设备状态
 * @irq: 要触发的中断位 (GPGPU_IRQ_xxx)
 */
static void gpgpu_raise_irq(GPGPUState *s, uint32_t irq)
{
    s->irq_status |= irq;
    gpgpu_update_irq(s);
}

/*
 * ============================================================================
 * 第二部分: DMA 引擎
 * ============================================================================
 * DMA (Direct Memory Access) 允许设备直接读写主机内存，
 * 无需 CPU 参与数据搬运。这是 GPU 进行大量数据传输的关键机制。
 *
 * 传输方向:
 * - Host -> VRAM: 上传数据到 GPU (如输入数组、内核代码)
 * - VRAM -> Host: 下载数据从 GPU (如计算结果)
 */

/**
 * gpgpu_dma_complete - DMA 传输完成回调
 * @opaque: GPGPUState 指针
 *
 * 当 DMA 定时器到期时调用，模拟 DMA 传输完成。
 * 实际的数据搬运在 gpgpu_dma_start 中已经完成，
 * 这里只是更新状态和触发中断。
 */
static void gpgpu_dma_complete(void *opaque)
{
    GPGPUState *s = GPGPU(opaque);

    /* 更新 DMA 状态 */
    s->dma.status &= ~GPGPU_DMA_BUSY;
    s->dma.status |= GPGPU_DMA_COMPLETE;

    /* 如果使能了中断，则触发 */
    if (s->dma.ctrl & GPGPU_DMA_IRQ_ENABLE) {
        gpgpu_raise_irq(s, GPGPU_IRQ_DMA_DONE);
    }
}

/**
 * gpgpu_dma_start - 启动 DMA 传输
 * @s: GPGPU 设备状态
 *
 * 执行实际的数据传输操作。
 * 使用 QEMU 的 pci_dma_read/write 函数来访问主机内存。
 */
static void gpgpu_dma_start(GPGPUState *s)
{
    uint64_t src = s->dma.src_addr;
    uint64_t dst = s->dma.dst_addr;
    uint32_t size = s->dma.size;
    bool to_vram = !(s->dma.ctrl & GPGPU_DMA_DIR_FROM_VRAM);

    /* 检查传输大小 */
    if (size == 0) {
        qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: DMA size is zero\n");
        s->dma.status |= GPGPU_DMA_ERROR;
        s->error_status |= GPGPU_ERR_DMA_FAULT;
        gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
        return;
    }

    /* 设置忙状态 */
    s->dma.status = GPGPU_DMA_BUSY;

    if (to_vram) {
        /*
         * Host -> VRAM 传输
         * src: 主机物理地址
         * dst: VRAM 偏移量
         */
        if (dst + size > s->vram_size) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU: DMA dst 0x%"PRIx64" + size %u exceeds VRAM\n",
                          dst, size);
            s->dma.status = GPGPU_DMA_ERROR;
            s->error_status |= GPGPU_ERR_DMA_FAULT;
            gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
            return;
        }

        /* 从主机内存读取数据到 VRAM */
        pci_dma_read(&s->parent_obj, src, s->vram_ptr + dst, size);
    } else {
        /*
         * VRAM -> Host 传输
         * src: VRAM 偏移量
         * dst: 主机物理地址
         */
        if (src + size > s->vram_size) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU: DMA src 0x%"PRIx64" + size %u exceeds VRAM\n",
                          src, size);
            s->dma.status = GPGPU_DMA_ERROR;
            s->error_status |= GPGPU_ERR_DMA_FAULT;
            gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
            return;
        }

        /* 从 VRAM 写入数据到主机内存 */
        pci_dma_write(&s->parent_obj, dst, s->vram_ptr + src, size);
    }

    /*
     * 使用定时器模拟 DMA 传输延迟
     * 实际硬件中 DMA 需要时间完成，这里模拟 1ms 延迟
     * TODO: 可以根据传输大小调整延迟时间
     */
    timer_mod(s->dma_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 1);
}

/*
 * ============================================================================
 * 第三部分: 内核执行模拟
 * ============================================================================
 * 当驱动向 DISPATCH 寄存器写入时，设备开始执行 GPU 内核。
 * 在真实硬件中，这会启动 SIMT 执行单元来并行执行线程。
 * 在模拟中，我们可以选择不同的实现策略:
 *
 * 1. 功能模拟: 使用定时器模拟延迟，不实际执行代码
 * 2. 精确模拟: 实际解释执行 RISC-V 指令 (需要集成 TCG 或解释器)
 *
 * 当前实现使用功能模拟方式。
 */

/**
 * gpgpu_kernel_complete - 内核执行完成回调
 * @opaque: GPGPUState 指针
 *
 * 当内核执行定时器到期时调用。
 */
static void gpgpu_kernel_complete(void *opaque)
{
    GPGPUState *s = GPGPU(opaque);

    /* 更新状态: 不再忙 */
    s->global_status &= ~GPGPU_STATUS_BUSY;

    /* 触发内核完成中断 */
    if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
        gpgpu_raise_irq(s, GPGPU_IRQ_KERNEL_DONE);
    }
}

/**
 * gpgpu_dispatch_kernel - 分发内核执行
 * @s: GPGPU 设备状态
 *
 * 验证内核参数并启动执行。
 */
static void gpgpu_dispatch_kernel(GPGPUState *s)
{
    /* 检查设备是否已使能 */
    if (!(s->global_ctrl & GPGPU_CTRL_ENABLE)) {
        qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: dispatch while disabled\n");
        s->error_status |= GPGPU_ERR_INVALID_CMD;
        gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
        return;
    }

    /* 检查设备是否忙 */
    if (s->global_status & GPGPU_STATUS_BUSY) {
        qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: dispatch while busy\n");
        s->error_status |= GPGPU_ERR_INVALID_CMD;
        gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
        return;
    }

    /* 验证 Grid 和 Block 维度 */
    if (s->kernel.grid_dim[0] == 0 || s->kernel.grid_dim[1] == 0 ||
        s->kernel.grid_dim[2] == 0) {
        qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: invalid grid dimensions\n");
        s->error_status |= GPGPU_ERR_INVALID_CMD;
        gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
        return;
    }

    if (s->kernel.block_dim[0] == 0 || s->kernel.block_dim[1] == 0 ||
        s->kernel.block_dim[2] == 0) {
        qemu_log_mask(LOG_GUEST_ERROR, "GPGPU: invalid block dimensions\n");
        s->error_status |= GPGPU_ERR_INVALID_CMD;
        gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
        return;
    }

    /* 验证内核地址在 VRAM 范围内 */
    if (s->kernel.kernel_addr >= s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU: kernel address 0x%"PRIx64" out of VRAM\n",
                      s->kernel.kernel_addr);
        s->error_status |= GPGPU_ERR_VRAM_FAULT;
        gpgpu_raise_irq(s, GPGPU_IRQ_ERROR);
        return;
    }

    /* 设置忙状态 */
    s->global_status |= GPGPU_STATUS_BUSY;

    /*
     * 计算模拟的执行时间
     * 简单模型: 时间与总线程数成正比
     * TODO: 可以实现更精确的时序模型
     */
    uint64_t total_threads = (uint64_t)s->kernel.grid_dim[0] *
                             s->kernel.grid_dim[1] *
                             s->kernel.grid_dim[2] *
                             s->kernel.block_dim[0] *
                             s->kernel.block_dim[1] *
                             s->kernel.block_dim[2];

    /* 每 1000 个线程增加 1ms 延迟，最少 1ms */
    uint64_t delay_ms = (total_threads / 1000) + 1;

    timer_mod(s->kernel_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + delay_ms);
}

/*
 * ============================================================================
 * 第四部分: MMIO 读写处理
 * ============================================================================
 * 当 CPU 访问 BAR 0 映射的地址时，QEMU 会调用这些回调函数。
 * 这是设备与软件交互的主要接口。
 */

/**
 * gpgpu_ctrl_read - 控制寄存器读操作
 * @opaque: GPGPUState 指针
 * @addr: 寄存器偏移地址 (相对于 BAR 0 起始地址)
 * @size: 访问大小 (字节)
 *
 * 返回: 寄存器值
 */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    uint64_t val = 0;

    switch (addr) {
    /*
     * 设备信息寄存器 (只读)
     * 驱动程序通过读取这些寄存器来识别设备和查询能力
     */
    case GPGPU_REG_DEV_ID:
        val = GPGPU_DEV_ID_VALUE;
        break;

    case GPGPU_REG_DEV_VERSION:
        val = GPGPU_DEV_VERSION_VALUE;
        break;

    case GPGPU_REG_DEV_CAPS:
        /*
         * 设备能力寄存器打包了多个配置值:
         * [7:0]   = 计算单元数量
         * [15:8]  = 每个 CU 的 warp 数量
         * [23:16] = warp 大小 (线程数)
         */
        val = (s->num_cus & 0xFF) |
              ((s->warps_per_cu & 0xFF) << 8) |
              ((s->warp_size & 0xFF) << 16);
        break;

    case GPGPU_REG_VRAM_SIZE_LO:
        val = (uint32_t)s->vram_size;
        break;

    case GPGPU_REG_VRAM_SIZE_HI:
        val = (uint32_t)(s->vram_size >> 32);
        break;

    /*
     * 全局控制寄存器
     */
    case GPGPU_REG_GLOBAL_CTRL:
        val = s->global_ctrl;
        break;

    case GPGPU_REG_GLOBAL_STATUS:
        val = s->global_status;
        break;

    case GPGPU_REG_ERROR_STATUS:
        val = s->error_status;
        break;

    /*
     * 中断控制寄存器
     */
    case GPGPU_REG_IRQ_ENABLE:
        val = s->irq_enable;
        break;

    case GPGPU_REG_IRQ_STATUS:
        val = s->irq_status;
        break;

    /*
     * 内核分发寄存器
     * 驱动可以读回之前配置的值
     */
    case GPGPU_REG_KERNEL_ADDR_LO:
        val = (uint32_t)s->kernel.kernel_addr;
        break;

    case GPGPU_REG_KERNEL_ADDR_HI:
        val = (uint32_t)(s->kernel.kernel_addr >> 32);
        break;

    case GPGPU_REG_KERNEL_ARGS_LO:
        val = (uint32_t)s->kernel.kernel_args;
        break;

    case GPGPU_REG_KERNEL_ARGS_HI:
        val = (uint32_t)(s->kernel.kernel_args >> 32);
        break;

    case GPGPU_REG_GRID_DIM_X:
        val = s->kernel.grid_dim[0];
        break;

    case GPGPU_REG_GRID_DIM_Y:
        val = s->kernel.grid_dim[1];
        break;

    case GPGPU_REG_GRID_DIM_Z:
        val = s->kernel.grid_dim[2];
        break;

    case GPGPU_REG_BLOCK_DIM_X:
        val = s->kernel.block_dim[0];
        break;

    case GPGPU_REG_BLOCK_DIM_Y:
        val = s->kernel.block_dim[1];
        break;

    case GPGPU_REG_BLOCK_DIM_Z:
        val = s->kernel.block_dim[2];
        break;

    case GPGPU_REG_SHARED_MEM_SIZE:
        val = s->kernel.shared_mem_size;
        break;

    /*
     * DMA 寄存器
     */
    case GPGPU_REG_DMA_SRC_LO:
        val = (uint32_t)s->dma.src_addr;
        break;

    case GPGPU_REG_DMA_SRC_HI:
        val = (uint32_t)(s->dma.src_addr >> 32);
        break;

    case GPGPU_REG_DMA_DST_LO:
        val = (uint32_t)s->dma.dst_addr;
        break;

    case GPGPU_REG_DMA_DST_HI:
        val = (uint32_t)(s->dma.dst_addr >> 32);
        break;

    case GPGPU_REG_DMA_SIZE:
        val = s->dma.size;
        break;

    case GPGPU_REG_DMA_CTRL:
        val = s->dma.ctrl;
        break;

    case GPGPU_REG_DMA_STATUS:
        val = s->dma.status;
        break;

    /*
     * 线程上下文寄存器
     * 在功能模拟模式下，这些返回 0
     * TODO: 在精确模拟模式下，需要返回当前执行线程的 ID
     */
    case GPGPU_REG_THREAD_ID_X:
    case GPGPU_REG_THREAD_ID_Y:
    case GPGPU_REG_THREAD_ID_Z:
    case GPGPU_REG_BLOCK_ID_X:
    case GPGPU_REG_BLOCK_ID_Y:
    case GPGPU_REG_BLOCK_ID_Z:
    case GPGPU_REG_WARP_ID:
    case GPGPU_REG_LANE_ID:
        val = 0;
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU: read from unknown register 0x%"HWADDR_PRIx"\n",
                      addr);
        val = 0;
        break;
    }

    return val;
}

/**
 * gpgpu_ctrl_write - 控制寄存器写操作
 * @opaque: GPGPUState 指针
 * @addr: 寄存器偏移地址
 * @val: 要写入的值
 * @size: 访问大小
 */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = GPGPU(opaque);

    switch (addr) {
    /*
     * 设备信息寄存器是只读的，写入被忽略
     */
    case GPGPU_REG_DEV_ID:
    case GPGPU_REG_DEV_VERSION:
    case GPGPU_REG_DEV_CAPS:
    case GPGPU_REG_VRAM_SIZE_LO:
    case GPGPU_REG_VRAM_SIZE_HI:
        /* 只读寄存器，忽略写入 */
        break;

    /*
     * 全局控制寄存器
     */
    case GPGPU_REG_GLOBAL_CTRL:
        s->global_ctrl = val;

        /* 处理软复位 */
        if (val & GPGPU_CTRL_RESET) {
            /* 复位设备状态 */
            s->global_status = GPGPU_STATUS_READY;
            s->error_status = 0;
            s->irq_status = 0;
            memset(&s->kernel, 0, sizeof(s->kernel));
            memset(&s->dma, 0, sizeof(s->dma));

            /* 取消所有挂起的定时器 */
            timer_del(s->dma_timer);
            timer_del(s->kernel_timer);

            /* 复位位自动清除 */
            s->global_ctrl &= ~GPGPU_CTRL_RESET;
        }

        /* 使能设备时设置就绪状态 */
        if (val & GPGPU_CTRL_ENABLE) {
            s->global_status |= GPGPU_STATUS_READY;
        } else {
            s->global_status &= ~GPGPU_STATUS_READY;
        }
        break;

    case GPGPU_REG_GLOBAL_STATUS:
        /* 只读寄存器 */
        break;

    case GPGPU_REG_ERROR_STATUS:
        /* 写 1 清除 (Write-1-to-Clear) */
        s->error_status &= ~val;
        /* 如果错误被清除，更新全局状态 */
        if (s->error_status == 0) {
            s->global_status &= ~GPGPU_STATUS_ERROR;
        }
        break;

    /*
     * 中断控制寄存器
     */
    case GPGPU_REG_IRQ_ENABLE:
        s->irq_enable = val;
        gpgpu_update_irq(s);
        break;

    case GPGPU_REG_IRQ_STATUS:
        /* 只读 */
        break;

    case GPGPU_REG_IRQ_ACK:
        /* 写 1 清除中断状态 */
        s->irq_status &= ~val;
        gpgpu_update_irq(s);
        break;

    /*
     * 内核分发寄存器
     */
    case GPGPU_REG_KERNEL_ADDR_LO:
        s->kernel.kernel_addr = (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) |
                                (val & 0xFFFFFFFF);
        break;

    case GPGPU_REG_KERNEL_ADDR_HI:
        s->kernel.kernel_addr = (s->kernel.kernel_addr & 0xFFFFFFFF) |
                                ((val & 0xFFFFFFFF) << 32);
        break;

    case GPGPU_REG_KERNEL_ARGS_LO:
        s->kernel.kernel_args = (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) |
                                (val & 0xFFFFFFFF);
        break;

    case GPGPU_REG_KERNEL_ARGS_HI:
        s->kernel.kernel_args = (s->kernel.kernel_args & 0xFFFFFFFF) |
                                ((val & 0xFFFFFFFF) << 32);
        break;

    case GPGPU_REG_GRID_DIM_X:
        s->kernel.grid_dim[0] = val;
        break;

    case GPGPU_REG_GRID_DIM_Y:
        s->kernel.grid_dim[1] = val;
        break;

    case GPGPU_REG_GRID_DIM_Z:
        s->kernel.grid_dim[2] = val;
        break;

    case GPGPU_REG_BLOCK_DIM_X:
        s->kernel.block_dim[0] = val;
        break;

    case GPGPU_REG_BLOCK_DIM_Y:
        s->kernel.block_dim[1] = val;
        break;

    case GPGPU_REG_BLOCK_DIM_Z:
        s->kernel.block_dim[2] = val;
        break;

    case GPGPU_REG_SHARED_MEM_SIZE:
        s->kernel.shared_mem_size = val;
        break;

    case GPGPU_REG_DISPATCH:
        /*
         * 触发内核执行！
         * 写入任意值都会启动内核分发
         */
        gpgpu_dispatch_kernel(s);
        break;

    /*
     * DMA 寄存器
     */
    case GPGPU_REG_DMA_SRC_LO:
        s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFF00000000ULL) |
                          (val & 0xFFFFFFFF);
        break;

    case GPGPU_REG_DMA_SRC_HI:
        s->dma.src_addr = (s->dma.src_addr & 0xFFFFFFFF) |
                          ((val & 0xFFFFFFFF) << 32);
        break;

    case GPGPU_REG_DMA_DST_LO:
        s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) |
                          (val & 0xFFFFFFFF);
        break;

    case GPGPU_REG_DMA_DST_HI:
        s->dma.dst_addr = (s->dma.dst_addr & 0xFFFFFFFF) |
                          ((val & 0xFFFFFFFF) << 32);
        break;

    case GPGPU_REG_DMA_SIZE:
        s->dma.size = val;
        break;

    case GPGPU_REG_DMA_CTRL:
        s->dma.ctrl = val;
        /* 如果设置了启动位，开始 DMA 传输 */
        if (val & GPGPU_DMA_START) {
            gpgpu_dma_start(s);
        }
        break;

    case GPGPU_REG_DMA_STATUS:
        /* 只读 */
        break;

    /*
     * 同步寄存器
     */
    case GPGPU_REG_BARRIER:
        /*
         * 在功能模拟模式下，屏障是空操作
         * TODO: 在精确模拟模式下实现线程同步
         */
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU: write to unknown register 0x%"HWADDR_PRIx"\n",
                      addr);
        break;
    }
}

/*
 * MMIO 操作结构
 * 定义了如何处理对 BAR 0 的访问
 */
static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,     /* 小端字节序 */
    .valid = {
        .min_access_size = 4,               /* 最小访问 4 字节 */
        .max_access_size = 4,               /* 最大访问 4 字节 */
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/*
 * ============================================================================
 * 第五部分: VRAM 访问
 * ============================================================================
 * BAR 2 映射显存，允许 CPU 直接读写 VRAM。
 * 这比 DMA 慢，但对于小量数据或调试很有用。
 */

static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = GPGPU(opaque);
    uint64_t val = 0;

    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU: VRAM read at 0x%"HWADDR_PRIx" out of bounds\n",
                      addr);
        return 0;
    }

    memcpy(&val, s->vram_ptr + addr, size);
    return val;
}

static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = GPGPU(opaque);

    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU: VRAM write at 0x%"HWADDR_PRIx" out of bounds\n",
                      addr);
        return;
    }

    memcpy(s->vram_ptr + addr, &val, size);
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

/*
 * ============================================================================
 * 第六部分: 门铃寄存器 (可选扩展)
 * ============================================================================
 * BAR 4 用于命令队列门铃，这是一个可选的高级功能。
 * 目前只是占位符实现。
 */

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    /* 门铃寄存器通常是只写的 */
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    /* TODO: 实现命令队列处理 */
    qemu_log_mask(LOG_UNIMP, "GPGPU: doorbell write not implemented\n");
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/*
 * ============================================================================
 * 第七部分: 设备生命周期管理
 * ============================================================================
 */

/**
 * gpgpu_realize - 设备实例化
 * @pdev: PCI 设备
 * @errp: 错误指针
 *
 * 当设备被创建时调用，负责初始化所有资源。
 * 这相当于设备的"构造函数"。
 */
static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    /*
     * 配置 PCI 中断引脚
     * INTA# = 1, INTB# = 2, INTC# = 3, INTD# = 4
     */
    pci_config_set_interrupt_pin(pci_conf, 1);

    /*
     * 初始化 MSI-X 中断
     * 参数: 设备, 向量数, BAR 索引, BAR 内偏移, BAR 索引, PBA 偏移, cap 偏移
     *
     * MSI-X 表和 PBA (Pending Bit Array) 放在 BAR 0 的高地址区域
     * 这样不会与控制寄存器冲突
     */
    if (msix_init_exclusive_bar(pdev, GPGPU_MSIX_VECTORS, 0, errp)) {
        return;
    }

    /* 初始化 MSI (作为 MSI-X 的后备) */
    msi_init(pdev, 0, 1, true, false, errp);

    /*
     * 分配显存
     * 使用 g_malloc0 分配并清零
     */
    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /*
     * 初始化内存区域并注册 BAR
     */

    /* BAR 0: 控制寄存器 */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: 显存 (64位, 可预取) */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: 门铃寄存器 (32位) */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    /*
     * 初始化定时器
     * 用于模拟异步操作 (DMA 和内核执行)
     */
    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_kernel_complete, s);

    /*
     * 设置初始状态
     */
    s->global_status = GPGPU_STATUS_READY;
}

/**
 * gpgpu_exit - 设备销毁
 * @pdev: PCI 设备
 *
 * 当设备被移除时调用，负责清理所有资源。
 */
static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    /* 删除定时器 */
    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);

    /* 释放显存 */
    g_free(s->vram_ptr);

    /* 清理中断 */
    msix_uninit_exclusive_bar(pdev);
    msi_uninit(pdev);
}

/**
 * gpgpu_reset - 设备复位
 * @dev: 设备对象
 *
 * 当设备复位时调用 (如系统重启)。
 */
static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    /* 复位所有寄存器状态 */
    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;

    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));

    /* 取消挂起的定时器 */
    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);

    /* 清零显存 */
    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

/*
 * ============================================================================
 * 第八部分: 设备属性和类初始化
 * ============================================================================
 */

/*
 * 设备属性定义
 * 这些属性可以通过 QEMU 命令行设置:
 *   -device gpgpu,num_cus=8,vram_size=128M
 */
static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

/*
 * 虚拟机状态迁移描述
 * 用于虚拟机快照和实时迁移
 */
static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        /* TODO: 添加更多状态字段 */
        VMSTATE_END_OF_LIST()
    }
};

/**
 * gpgpu_class_init - 设备类初始化
 * @klass: 类对象
 * @data: 用户数据
 *
 * 设置 PCI 设备类的各种回调函数和属性。
 */
static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    /* PCI 设备回调 */
    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;

    /* PCI ID 配置 */
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    /* 设备类配置 */
    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);

    /* 设备分类 */
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

/*
 * QOM 类型信息
 * 向 QEMU 的对象系统注册设备类型
 */
static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },  /* 声明这是一个 PCIe 设备 */
        { }
    },
};

/**
 * gpgpu_register_types - 注册设备类型
 *
 * QEMU 启动时自动调用，将设备类型注册到对象系统。
 */
static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
