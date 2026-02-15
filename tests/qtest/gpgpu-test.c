/*
 * QTest testcase for GPGPU device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/module.h"
#include "libqtest.h"
#include "libqos/qgraph.h"
#include "libqos/pci.h"

/*
 * GPGPU 设备寄存器定义 (从 hw/gpgpu/gpgpu.h 复制)
 * 为了测试独立性，这里重新定义需要的常量
 */

/* PCI 配置 */
#define GPGPU_VENDOR_ID         0x1234
#define GPGPU_DEVICE_ID         0x1337

/* 设备信息寄存器 (BAR0) */
#define GPGPU_REG_DEV_ID            0x0000
#define GPGPU_REG_DEV_VERSION       0x0004
#define GPGPU_REG_DEV_CAPS          0x0008
#define GPGPU_REG_VRAM_SIZE_LO      0x000C
#define GPGPU_REG_VRAM_SIZE_HI      0x0010

/* 全局控制寄存器 */
#define GPGPU_REG_GLOBAL_CTRL       0x0100
#define GPGPU_REG_GLOBAL_STATUS     0x0104
#define GPGPU_REG_ERROR_STATUS      0x0108

/* 中断寄存器 */
#define GPGPU_REG_IRQ_ENABLE        0x0200
#define GPGPU_REG_IRQ_STATUS        0x0204
#define GPGPU_REG_IRQ_ACK           0x0208

/* 内核分发寄存器 */
#define GPGPU_REG_GRID_DIM_X        0x0310
#define GPGPU_REG_GRID_DIM_Y        0x0314
#define GPGPU_REG_GRID_DIM_Z        0x0318
#define GPGPU_REG_BLOCK_DIM_X       0x031C
#define GPGPU_REG_BLOCK_DIM_Y       0x0320
#define GPGPU_REG_BLOCK_DIM_Z       0x0324

/* DMA 寄存器 */
#define GPGPU_REG_DMA_SRC_LO        0x0400
#define GPGPU_REG_DMA_SRC_HI        0x0404
#define GPGPU_REG_DMA_DST_LO        0x0408
#define GPGPU_REG_DMA_DST_HI        0x040C
#define GPGPU_REG_DMA_SIZE          0x0410
#define GPGPU_REG_DMA_CTRL          0x0414
#define GPGPU_REG_DMA_STATUS        0x0418

/* 寄存器位定义 */
#define GPGPU_CTRL_ENABLE           (1 << 0)
#define GPGPU_CTRL_RESET            (1 << 1)
#define GPGPU_STATUS_READY          (1 << 0)
#define GPGPU_DMA_START             (1 << 0)
#define GPGPU_DMA_DIR_TO_VRAM       (0 << 1)
#define GPGPU_DMA_DIR_FROM_VRAM     (1 << 1)
#define GPGPU_DMA_COMPLETE          (1 << 1)

/* 设备标识值 */
#define GPGPU_DEV_ID_VALUE          0x47505055  /* "GPPU" */
#define GPGPU_DEV_VERSION_VALUE     0x00010000  /* v1.0.0 */

/* 默认配置 */
#define GPGPU_DEFAULT_VRAM_SIZE     (64 * 1024 * 1024)

typedef struct QGPGPU {
    QOSGraphObject obj;
    QPCIDevice dev;
    QPCIBar bar0;   /* 控制寄存器 */
    QPCIBar bar2;   /* VRAM */
} QGPGPU;

static void *gpgpu_get_driver(void *obj, const char *interface)
{
    QGPGPU *gpgpu = obj;

    if (!g_strcmp0(interface, "pci-device")) {
        return &gpgpu->dev;
    }

    fprintf(stderr, "%s not present in gpgpu\n", interface);
    g_assert_not_reached();
}

static void *gpgpu_create(void *pci_bus, QGuestAllocator *alloc, void *addr)
{
    QGPGPU *gpgpu = g_new0(QGPGPU, 1);
    QPCIBus *bus = pci_bus;

    qpci_device_init(&gpgpu->dev, bus, addr);
    gpgpu->obj.get_driver = gpgpu_get_driver;

    return &gpgpu->obj;
}

/*
 * 测试 1: 设备识别测试
 * 验证设备 ID 和版本号寄存器返回正确的值
 */
static void gpgpu_test_device_id(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t dev_id, version;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 读取设备 ID */
    dev_id = qpci_io_readl(pdev, bar0, GPGPU_REG_DEV_ID);
    g_assert_cmphex(dev_id, ==, GPGPU_DEV_ID_VALUE);

    /* 读取版本号 */
    version = qpci_io_readl(pdev, bar0, GPGPU_REG_DEV_VERSION);
    g_assert_cmphex(version, ==, GPGPU_DEV_VERSION_VALUE);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 2: VRAM 大小寄存器测试
 * 验证显存大小寄存器返回配置的值
 */
static void gpgpu_test_vram_size(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t vram_lo, vram_hi;
    uint64_t vram_size;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 读取 VRAM 大小 */
    vram_lo = qpci_io_readl(pdev, bar0, GPGPU_REG_VRAM_SIZE_LO);
    vram_hi = qpci_io_readl(pdev, bar0, GPGPU_REG_VRAM_SIZE_HI);
    vram_size = ((uint64_t)vram_hi << 32) | vram_lo;

    g_assert_cmpuint(vram_size, ==, GPGPU_DEFAULT_VRAM_SIZE);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 3: 全局控制寄存器测试
 * 验证设备使能和状态寄存器工作正常
 */
static void gpgpu_test_global_ctrl(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t ctrl, status;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 读取初始状态 */
    status = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_STATUS);
    g_assert_cmpuint(status & GPGPU_STATUS_READY, ==, GPGPU_STATUS_READY);

    /* 测试写入控制寄存器 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);
    ctrl = qpci_io_readl(pdev, bar0, GPGPU_REG_GLOBAL_CTRL);
    g_assert_cmpuint(ctrl & GPGPU_CTRL_ENABLE, ==, GPGPU_CTRL_ENABLE);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 4: Grid/Block 维度寄存器测试
 * 验证 kernel dispatch 配置寄存器可读写
 */
static void gpgpu_test_dispatch_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 写入 Grid 维度 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_X, 64);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Y, 32);
    qpci_io_writel(pdev, bar0, GPGPU_REG_GRID_DIM_Z, 1);

    /* 验证读回 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GRID_DIM_X);
    g_assert_cmpuint(val, ==, 64);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GRID_DIM_Y);
    g_assert_cmpuint(val, ==, 32);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_GRID_DIM_Z);
    g_assert_cmpuint(val, ==, 1);

    /* 写入 Block 维度 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_X, 256);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Y, 1);
    qpci_io_writel(pdev, bar0, GPGPU_REG_BLOCK_DIM_Z, 1);

    /* 验证读回 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_BLOCK_DIM_X);
    g_assert_cmpuint(val, ==, 256);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 5: VRAM 读写测试
 * 验证显存区域 (BAR2) 可正确读写
 */
static void gpgpu_test_vram_access(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar2;
    uint32_t test_pattern = 0xDEADBEEF;
    uint32_t read_val;

    qpci_device_enable(pdev);
    bar2 = qpci_iomap(pdev, 2, NULL);

    /* 写入测试数据 */
    qpci_io_writel(pdev, bar2, 0x0, test_pattern);
    qpci_io_writel(pdev, bar2, 0x100, 0x12345678);
    qpci_io_writel(pdev, bar2, 0x1000, 0xCAFEBABE);

    /* 验证读回 */
    read_val = qpci_io_readl(pdev, bar2, 0x0);
    g_assert_cmphex(read_val, ==, test_pattern);

    read_val = qpci_io_readl(pdev, bar2, 0x100);
    g_assert_cmphex(read_val, ==, 0x12345678);

    read_val = qpci_io_readl(pdev, bar2, 0x1000);
    g_assert_cmphex(read_val, ==, 0xCAFEBABE);

    qpci_iounmap(pdev, bar2);
}

/*
 * 测试 6: DMA 寄存器测试
 * 验证 DMA 控制寄存器可正确配置
 */
static void gpgpu_test_dma_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 配置 DMA 源地址 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_SRC_LO, 0x1000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_SRC_HI, 0x0);

    /* 配置 DMA 目标地址 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_DST_LO, 0x2000);
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_DST_HI, 0x0);

    /* 配置传输大小 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_DMA_SIZE, 4096);

    /* 验证读回 */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_DMA_SRC_LO);
    g_assert_cmphex(val, ==, 0x1000);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_DMA_DST_LO);
    g_assert_cmphex(val, ==, 0x2000);

    val = qpci_io_readl(pdev, bar0, GPGPU_REG_DMA_SIZE);
    g_assert_cmpuint(val, ==, 4096);

    qpci_iounmap(pdev, bar0);
}

/*
 * 测试 7: 中断寄存器测试
 * 验证中断使能和状态寄存器可正确操作
 */
static void gpgpu_test_irq_regs(void *obj, void *data, QGuestAllocator *alloc)
{
    QGPGPU *gpgpu = obj;
    QPCIDevice *pdev = &gpgpu->dev;
    QPCIBar bar0;
    uint32_t val;

    qpci_device_enable(pdev);
    bar0 = qpci_iomap(pdev, 0, NULL);

    /* 写入中断使能 */
    qpci_io_writel(pdev, bar0, GPGPU_REG_IRQ_ENABLE, 0x7);
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_IRQ_ENABLE);
    g_assert_cmphex(val, ==, 0x7);

    /* 读取中断状态 (应该为 0) */
    val = qpci_io_readl(pdev, bar0, GPGPU_REG_IRQ_STATUS);
    g_assert_cmphex(val, ==, 0x0);

    qpci_iounmap(pdev, bar0);
}

static void gpgpu_register_nodes(void)
{
    QOSGraphEdgeOptions opts = {
        .extra_device_opts = "addr=04.0",
    };

    add_qpci_address(&opts, &(QPCIAddress) {
        .devfn = QPCI_DEVFN(4, 0),
        .vendor_id = GPGPU_VENDOR_ID,
        .device_id = GPGPU_DEVICE_ID,
    });

    /* 创建驱动节点 */
    qos_node_create_driver("gpgpu", gpgpu_create);
    qos_node_consumes("gpgpu", "pci-bus", &opts);
    qos_node_produces("gpgpu", "pci-device");

    /* 注册测试用例 */
    qos_add_test("device-id", "gpgpu", gpgpu_test_device_id, NULL);
    qos_add_test("vram-size", "gpgpu", gpgpu_test_vram_size, NULL);
    qos_add_test("global-ctrl", "gpgpu", gpgpu_test_global_ctrl, NULL);
    qos_add_test("dispatch-regs", "gpgpu", gpgpu_test_dispatch_regs, NULL);
    qos_add_test("vram-access", "gpgpu", gpgpu_test_vram_access, NULL);
    qos_add_test("dma-regs", "gpgpu", gpgpu_test_dma_regs, NULL);
    qos_add_test("irq-regs", "gpgpu", gpgpu_test_irq_regs, NULL);
}

libqos_init(gpgpu_register_nodes);
