/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 *
 * 简化的 RV32I 指令解释器实现。
 * 支持的指令集: RV32I 子集 (约 25 条指令)
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/*
 * ============================================================================
 * RV32I 指令编码常量
 * ============================================================================
 */

/* Opcode (inst[6:0]) */
#define OPCODE_LUI      0x37    /* 0110111 */
#define OPCODE_AUIPC    0x17    /* 0010111 */
#define OPCODE_JAL      0x6F    /* 1101111 */
#define OPCODE_JALR     0x67    /* 1100111 */
#define OPCODE_BRANCH   0x63    /* 1100011 */
#define OPCODE_LOAD     0x03    /* 0000011 */
#define OPCODE_STORE    0x23    /* 0100011 */
#define OPCODE_OP_IMM   0x13    /* 0010011 */
#define OPCODE_OP       0x33    /* 0110011 */
#define OPCODE_SYSTEM   0x73    /* 1110011 */

/* funct3 for OP and OP_IMM */
#define FUNCT3_ADD_SUB  0x0
#define FUNCT3_SLL      0x1
#define FUNCT3_SLT      0x2
#define FUNCT3_SLTU     0x3
#define FUNCT3_XOR      0x4
#define FUNCT3_SRL_SRA  0x5
#define FUNCT3_OR       0x6
#define FUNCT3_AND      0x7

/* funct3 for BRANCH */
#define FUNCT3_BEQ      0x0
#define FUNCT3_BNE      0x1
#define FUNCT3_BLT      0x4
#define FUNCT3_BGE      0x5
#define FUNCT3_BLTU     0x6
#define FUNCT3_BGEU     0x7

/* funct3 for LOAD */
#define FUNCT3_LB       0x0
#define FUNCT3_LH       0x1
#define FUNCT3_LW       0x2
#define FUNCT3_LBU      0x4
#define FUNCT3_LHU      0x5

/* funct3 for STORE */
#define FUNCT3_SB       0x0
#define FUNCT3_SH       0x1
#define FUNCT3_SW       0x2

/*
 * ============================================================================
 * 指令解码辅助宏
 * ============================================================================
 */
#define BITS(x, hi, lo) (((x) >> (lo)) & ((1U << ((hi) - (lo) + 1)) - 1))
#define SEXT(x, len) (((int32_t)((x) << (32 - (len)))) >> (32 - (len)))

#define GET_OPCODE(inst)    ((inst) & 0x7F)
#define GET_RD(inst)        BITS(inst, 11, 7)
#define GET_FUNCT3(inst)    BITS(inst, 14, 12)
#define GET_RS1(inst)       BITS(inst, 19, 15)
#define GET_RS2(inst)       BITS(inst, 24, 20)
#define GET_FUNCT7(inst)    BITS(inst, 31, 25)

/* I-type 立即数 */
#define GET_IMM_I(inst)     SEXT(BITS(inst, 31, 20), 12)

/* S-type 立即数 */
#define GET_IMM_S(inst)     SEXT((BITS(inst, 31, 25) << 5) | BITS(inst, 11, 7), 12)

/* B-type 立即数 */
#define GET_IMM_B(inst)     SEXT((BITS(inst, 31, 31) << 12) | \
                                 (BITS(inst, 7, 7) << 11) | \
                                 (BITS(inst, 30, 25) << 5) | \
                                 (BITS(inst, 11, 8) << 1), 13)

/* U-type 立即数 */
#define GET_IMM_U(inst)     ((inst) & 0xFFFFF000)

/* J-type 立即数 */
#define GET_IMM_J(inst)     SEXT((BITS(inst, 31, 31) << 20) | \
                                 (BITS(inst, 19, 12) << 12) | \
                                 (BITS(inst, 20, 20) << 11) | \
                                 (BITS(inst, 30, 21) << 1), 21)

/*
 * ============================================================================
 * 内存访问函数
 * ============================================================================
 */

/* 当前执行上下文（用于内存访问时获取 lane 信息） */
static __thread GPGPUWarp *current_warp;
static __thread int current_lane_id;

/**
 * core_mem_read - GPU 核心内存读取
 * @s: GPGPU 设备状态
 * @addr: 地址
 * @size: 大小 (1, 2, 4)
 *
 * 返回: 读取的值
 */
static uint32_t core_mem_read(GPGPUState *s, uint32_t addr, int size)
{
    /* 检查是否是 CTRL 寄存器访问 */
    if (addr >= GPGPU_CORE_CTRL_BASE &&
        addr < GPGPU_CORE_CTRL_BASE + 0x100) {
        switch (addr) {
        case GPGPU_CORE_CTRL_THREAD_ID_X:
            return current_lane_id;  /* 简化：只支持 1D */
        case GPGPU_CORE_CTRL_THREAD_ID_Y:
            return 0;
        case GPGPU_CORE_CTRL_THREAD_ID_Z:
            return 0;
        case GPGPU_CORE_CTRL_BLOCK_ID_X:
            return current_warp->block_id[0];
        case GPGPU_CORE_CTRL_BLOCK_ID_Y:
            return current_warp->block_id[1];
        case GPGPU_CORE_CTRL_BLOCK_ID_Z:
            return current_warp->block_id[2];
        case GPGPU_CORE_CTRL_BLOCK_DIM_X:
            return s->kernel.block_dim[0];
        case GPGPU_CORE_CTRL_BLOCK_DIM_Y:
            return s->kernel.block_dim[1];
        case GPGPU_CORE_CTRL_BLOCK_DIM_Z:
            return s->kernel.block_dim[2];
        case GPGPU_CORE_CTRL_GRID_DIM_X:
            return s->kernel.grid_dim[0];
        case GPGPU_CORE_CTRL_GRID_DIM_Y:
            return s->kernel.grid_dim[1];
        case GPGPU_CORE_CTRL_GRID_DIM_Z:
            return s->kernel.grid_dim[2];
        default:
            return 0;
        }
    }

    /* VRAM 访问 */
    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU core: read addr 0x%x out of VRAM bounds\n", addr);
        return 0;
    }

    uint32_t val = 0;
    memcpy(&val, s->vram_ptr + addr, size);
    return val;
}

/**
 * core_mem_write - GPU 核心内存写入
 * @s: GPGPU 设备状态
 * @addr: 地址
 * @val: 要写入的值
 * @size: 大小 (1, 2, 4)
 */
static void core_mem_write(GPGPUState *s, uint32_t addr, uint32_t val, int size)
{
    /* CTRL 寄存器是只读的（对于 GPU 核心） */
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return;
    }

    /* VRAM 访问 */
    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU core: write addr 0x%x out of VRAM bounds\n", addr);
        return;
    }

    memcpy(s->vram_ptr + addr, &val, size);
}

/*
 * ============================================================================
 * 指令执行
 * ============================================================================
 */

/**
 * exec_one_inst - 执行一条指令
 * @s: GPGPU 设备状态
 * @lane: lane 状态
 * @inst: 指令
 *
 * 返回: 0 继续执行, 1 停止 (ebreak/ret), -1 错误
 */
static int exec_one_inst(GPGPUState *s, GPGPULane *lane, uint32_t inst)
{
    uint32_t opcode = GET_OPCODE(inst);
    uint32_t rd = GET_RD(inst);
    uint32_t rs1 = GET_RS1(inst);
    uint32_t rs2 = GET_RS2(inst);
    uint32_t funct3 = GET_FUNCT3(inst);
    uint32_t funct7 = GET_FUNCT7(inst);

    uint32_t src1 = lane->gpr[rs1];
    uint32_t src2 = lane->gpr[rs2];
    uint32_t next_pc = lane->pc + 4;

    switch (opcode) {
    case OPCODE_LUI:
        lane->gpr[rd] = GET_IMM_U(inst);
        break;

    case OPCODE_AUIPC:
        lane->gpr[rd] = lane->pc + GET_IMM_U(inst);
        break;

    case OPCODE_JAL:
        lane->gpr[rd] = lane->pc + 4;
        next_pc = lane->pc + GET_IMM_J(inst);
        break;

    case OPCODE_JALR:
        lane->gpr[rd] = lane->pc + 4;
        next_pc = (src1 + GET_IMM_I(inst)) & ~1U;
        /* 检查是否是 ret (jalr x0, ra, 0) */
        if (rd == 0 && rs1 == 1 && GET_IMM_I(inst) == 0) {
            /* ret 指令，如果 ra == 0 则停止执行 */
            if (lane->gpr[1] == 0) {
                return 1;  /* 停止 */
            }
        }
        break;

    case OPCODE_BRANCH: {
        int32_t imm = GET_IMM_B(inst);
        bool taken = false;
        switch (funct3) {
        case FUNCT3_BEQ:  taken = (src1 == src2); break;
        case FUNCT3_BNE:  taken = (src1 != src2); break;
        case FUNCT3_BLT:  taken = ((int32_t)src1 < (int32_t)src2); break;
        case FUNCT3_BGE:  taken = ((int32_t)src1 >= (int32_t)src2); break;
        case FUNCT3_BLTU: taken = (src1 < src2); break;
        case FUNCT3_BGEU: taken = (src1 >= src2); break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU core: unknown branch funct3 %d\n", funct3);
            return -1;
        }
        if (taken) {
            next_pc = lane->pc + imm;
        }
        break;
    }

    case OPCODE_LOAD: {
        int32_t imm = GET_IMM_I(inst);
        uint32_t addr = src1 + imm;
        uint32_t val = 0;
        switch (funct3) {
        case FUNCT3_LB:
            val = (int8_t)core_mem_read(s, addr, 1);
            break;
        case FUNCT3_LH:
            val = (int16_t)core_mem_read(s, addr, 2);
            break;
        case FUNCT3_LW:
            val = core_mem_read(s, addr, 4);
            break;
        case FUNCT3_LBU:
            val = (uint8_t)core_mem_read(s, addr, 1);
            break;
        case FUNCT3_LHU:
            val = (uint16_t)core_mem_read(s, addr, 2);
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU core: unknown load funct3 %d\n", funct3);
            return -1;
        }
        lane->gpr[rd] = val;
        break;
    }

    case OPCODE_STORE: {
        int32_t imm = GET_IMM_S(inst);
        uint32_t addr = src1 + imm;
        switch (funct3) {
        case FUNCT3_SB:
            core_mem_write(s, addr, src2, 1);
            break;
        case FUNCT3_SH:
            core_mem_write(s, addr, src2, 2);
            break;
        case FUNCT3_SW:
            core_mem_write(s, addr, src2, 4);
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU core: unknown store funct3 %d\n", funct3);
            return -1;
        }
        break;
    }

    case OPCODE_OP_IMM: {
        int32_t imm = GET_IMM_I(inst);
        uint32_t shamt = imm & 0x1F;
        uint32_t result = 0;
        switch (funct3) {
        case FUNCT3_ADD_SUB:  /* addi */
            result = src1 + imm;
            break;
        case FUNCT3_SLT:      /* slti */
            result = ((int32_t)src1 < imm) ? 1 : 0;
            break;
        case FUNCT3_SLTU:     /* sltiu */
            result = (src1 < (uint32_t)imm) ? 1 : 0;
            break;
        case FUNCT3_XOR:      /* xori */
            result = src1 ^ imm;
            break;
        case FUNCT3_OR:       /* ori */
            result = src1 | imm;
            break;
        case FUNCT3_AND:      /* andi */
            result = src1 & imm;
            break;
        case FUNCT3_SLL:      /* slli */
            result = src1 << shamt;
            break;
        case FUNCT3_SRL_SRA:  /* srli / srai */
            if (funct7 & 0x20) {
                result = (int32_t)src1 >> shamt;  /* srai */
            } else {
                result = src1 >> shamt;           /* srli */
            }
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU core: unknown OP_IMM funct3 %d\n", funct3);
            return -1;
        }
        lane->gpr[rd] = result;
        break;
    }

    case OPCODE_OP: {
        uint32_t result = 0;
        switch (funct3) {
        case FUNCT3_ADD_SUB:
            if (funct7 & 0x20) {
                result = src1 - src2;  /* sub */
            } else {
                result = src1 + src2;  /* add */
            }
            break;
        case FUNCT3_SLL:
            result = src1 << (src2 & 0x1F);
            break;
        case FUNCT3_SLT:
            result = ((int32_t)src1 < (int32_t)src2) ? 1 : 0;
            break;
        case FUNCT3_SLTU:
            result = (src1 < src2) ? 1 : 0;
            break;
        case FUNCT3_XOR:
            result = src1 ^ src2;
            break;
        case FUNCT3_SRL_SRA:
            if (funct7 & 0x20) {
                result = (int32_t)src1 >> (src2 & 0x1F);  /* sra */
            } else {
                result = src1 >> (src2 & 0x1F);           /* srl */
            }
            break;
        case FUNCT3_OR:
            result = src1 | src2;
            break;
        case FUNCT3_AND:
            result = src1 & src2;
            break;
        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "GPGPU core: unknown OP funct3 %d\n", funct3);
            return -1;
        }
        lane->gpr[rd] = result;
        break;
    }

    case OPCODE_SYSTEM:
        /* ebreak: 停止执行 */
        if (inst == 0x00100073) {
            return 1;
        }
        /* 其他 system 指令忽略 */
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU core: unknown opcode 0x%x (inst=0x%08x)\n",
                      opcode, inst);
        return -1;
    }

    /* x0 恒为 0 */
    lane->gpr[0] = 0;

    /* 更新 PC */
    lane->pc = next_pc;

    return 0;
}

/*
 * ============================================================================
 * Warp 执行
 * ============================================================================
 */

void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads)
{
    memset(warp, 0, sizeof(*warp));

    warp->thread_id_base = thread_id_base;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    /* 初始化每个 lane */
    for (uint32_t i = 0; i < GPGPU_WARP_SIZE; i++) {
        if (i < num_threads) {
            warp->lanes[i].pc = pc;
            warp->lanes[i].active = true;
            warp->active_mask |= (1U << i);
        } else {
            warp->lanes[i].active = false;
        }
        /* 寄存器初始化为 0 */
        memset(warp->lanes[i].gpr, 0, sizeof(warp->lanes[i].gpr));
    }
}

int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    current_warp = warp;
    uint32_t cycles = 0;

    while (warp->active_mask && cycles < max_cycles) {
        /* 取指令 (所有活跃 lane 执行同一条指令，使用 lane 0 的 PC) */
        int first_active = __builtin_ctz(warp->active_mask);
        uint32_t pc = warp->lanes[first_active].pc;
        uint32_t inst = core_mem_read(s, pc, 4);

        /* 对每个活跃 lane 执行指令 */
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (!(warp->active_mask & (1U << i))) {
                continue;
            }

            current_lane_id = warp->thread_id_base + i;
            GPGPULane *lane = &warp->lanes[i];

            int ret = exec_one_inst(s, lane, inst);
            if (ret < 0) {
                /* 错误 */
                return -1;
            } else if (ret > 0) {
                /* 线程结束 */
                lane->active = false;
                warp->active_mask &= ~(1U << i);
            }
        }

        cycles++;
    }

    if (cycles >= max_cycles) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "GPGPU core: warp exceeded max cycles (%u)\n", max_cycles);
        return -1;
    }

    return 0;
}

/*
 * ============================================================================
 * Kernel 执行
 * ============================================================================
 */

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_x = s->kernel.grid_dim[0];
    uint32_t grid_y = s->kernel.grid_dim[1];
    uint32_t grid_z = s->kernel.grid_dim[2];
    uint32_t block_x = s->kernel.block_dim[0];
    uint32_t block_y = s->kernel.block_dim[1];
    uint32_t block_z = s->kernel.block_dim[2];

    uint32_t threads_per_block = block_x * block_y * block_z;
    uint32_t kernel_addr = (uint32_t)s->kernel.kernel_addr;

    /* 遍历所有 block */
    for (uint32_t bz = 0; bz < grid_z; bz++) {
        for (uint32_t by = 0; by < grid_y; by++) {
            for (uint32_t bx = 0; bx < grid_x; bx++) {
                uint32_t block_id[3] = {bx, by, bz};

                /* 每个 block 分成多个 warp 执行 */
                for (uint32_t tid = 0; tid < threads_per_block;
                     tid += GPGPU_WARP_SIZE) {
                    GPGPUWarp warp;
                    uint32_t num_threads = threads_per_block - tid;
                    if (num_threads > GPGPU_WARP_SIZE) {
                        num_threads = GPGPU_WARP_SIZE;
                    }

                    gpgpu_core_init_warp(&warp, kernel_addr, tid, block_id,
                                         num_threads);

                    int ret = gpgpu_core_exec_warp(s, &warp, 100000);
                    if (ret < 0) {
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}
