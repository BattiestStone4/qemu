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

/* RV32F Opcodes */
#define OPCODE_LOAD_FP  0x07    /* 0000111 */
#define OPCODE_STORE_FP 0x27    /* 0100111 */
#define OPCODE_FMADD    0x43    /* 1000011 */
#define OPCODE_FMSUB    0x47    /* 1000111 */
#define OPCODE_FNMSUB   0x4B    /* 1001011 */
#define OPCODE_FNMADD   0x4F    /* 1001111 */
#define OPCODE_OP_FP    0x53    /* 1010011 */

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

/* OP_FP funct7 constants */
#define FUNCT7_FADD_S     0x00
#define FUNCT7_FSUB_S     0x04
#define FUNCT7_FMUL_S     0x08
#define FUNCT7_FDIV_S     0x0C
#define FUNCT7_FSQRT_S    0x2C
#define FUNCT7_FSGNJ_S    0x10
#define FUNCT7_FMINMAX_S  0x14
#define FUNCT7_FCVT_W_S   0x60
#define FUNCT7_FCVT_S_W   0x68
#define FUNCT7_FMV_X_W    0x70
#define FUNCT7_FCMP_S     0x50
#define FUNCT7_FMV_W_X    0x78

/* Low-precision float conversion funct7 (custom extension) */
#define FUNCT7_FCVT_BF16  0x22
#define FUNCT7_FCVT_FP8   0x24
#define FUNCT7_FCVT_FP4   0x26

/* funct3 for SYSTEM (CSR instructions) */
#define FUNCT3_CSRRW    0x1
#define FUNCT3_CSRRS    0x2
#define FUNCT3_CSRRC    0x3
#define FUNCT3_CSRRWI   0x5
#define FUNCT3_CSRRSI   0x6
#define FUNCT3_CSRRCI   0x7

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
#define GET_CSR(inst)       BITS(inst, 31, 20)

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

/* R4-type (fused multiply-add) */
#define GET_RS3(inst)       BITS(inst, 31, 27)
#define GET_FUNCT2(inst)    BITS(inst, 26, 25)

/*
 * ============================================================================
 * 内存访问函数
 * ============================================================================
 */

/* 当前执行上下文（用于内存访问时获取 lane 信息） */
static __thread GPGPUWarp *current_warp;
static __thread GPGPULane *current_lane;

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
            return MHARTID_THREAD(current_lane->mhartid);
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
 * RV32F 辅助函数
 * ============================================================================
 */

/**
 * decode_rm - 解码舍入模式
 * rm 字段 (funct3): 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM, 7=dynamic(fcsr.frm)
 */
static FloatRoundMode decode_rm(uint32_t rm, uint32_t fcsr)
{
    if (rm == 7) {
        rm = (fcsr >> 5) & 0x7;
    }
    switch (rm) {
    case 0: return float_round_nearest_even;
    case 1: return float_round_to_zero;
    case 2: return float_round_down;
    case 3: return float_round_up;
    case 4: return float_round_nearest_even; /* RMM: approx as RNE */
    default: return float_round_nearest_even;
    }
}

/**
 * prep_fp_status - 设置舍入模式并清除异常标志
 */
static void prep_fp_status(float_status *fps, FloatRoundMode rm)
{
    set_float_rounding_mode(rm, fps);
    set_float_exception_flags(0, fps);
}

/**
 * sync_fp_exceptions - 将 softfloat 异常标志累积到 fcsr.fflags
 * RISC-V fflags: bit0=NX, bit1=UF, bit2=OF, bit3=DZ, bit4=NV
 */
static void sync_fp_exceptions(GPGPULane *lane)
{
    int sf_flags = get_float_exception_flags(&lane->fp_status);
    uint32_t rv_flags = 0;

    if (sf_flags & float_flag_inexact) {
        rv_flags |= (1 << 0);  /* NX */
    }
    if (sf_flags & float_flag_underflow) {
        rv_flags |= (1 << 1);  /* UF */
    }
    if (sf_flags & float_flag_overflow) {
        rv_flags |= (1 << 2);  /* OF */
    }
    if (sf_flags & float_flag_divbyzero) {
        rv_flags |= (1 << 3);  /* DZ */
    }
    if (sf_flags & float_flag_invalid) {
        rv_flags |= (1 << 4);  /* NV */
    }
    lane->fcsr |= rv_flags;  /* OR 累积 */
}

/**
 * fclass_s - RISC-V fclass.s 分类逻辑
 * 返回 10 位掩码中设置一位
 */
static uint32_t fclass_s(float32 val, float_status *fps)
{
    bool is_neg = float32_is_neg(val);
    bool is_inf = float32_is_infinity(val);
    bool is_zero = float32_is_zero(val);
    bool is_snan = float32_is_signaling_nan(val, fps);
    bool is_qnan = float32_is_quiet_nan(val, fps);
    bool is_denorm = float32_is_zero_or_denormal(val) && !is_zero;
    bool is_normal = float32_is_normal(val);

    if (is_neg && is_inf)    return 1 << 0;   /* -inf */
    if (is_neg && is_normal) return 1 << 1;   /* -normal */
    if (is_neg && is_denorm) return 1 << 2;   /* -subnormal */
    if (is_neg && is_zero)   return 1 << 3;   /* -0 */
    if (!is_neg && is_zero)  return 1 << 4;   /* +0 */
    if (!is_neg && is_denorm) return 1 << 5;  /* +subnormal */
    if (!is_neg && is_normal) return 1 << 6;  /* +normal */
    if (!is_neg && is_inf)   return 1 << 7;   /* +inf */
    if (is_snan)             return 1 << 8;   /* signaling NaN */
    if (is_qnan)             return 1 << 9;   /* quiet NaN */
    return 1 << 9; /* fallback: qNaN */
}

/**
 * float32_to_float4_e2m1 - FP32 → E2M1 (4-bit float) conversion
 *
 * E2M1 format: sign(1) + exp(2, bias=1) + man(1) = 4 bits
 * Representable magnitudes: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
 * Values beyond 6.0 are saturated.  No Inf/NaN representation.
 */
static float4_e2m1 float32_to_float4_e2m1(float32 a, float_status *status)
{
    uint32_t sign = (a >> 31) & 1;
    uint32_t abs_a = a & 0x7FFFFFFF;

    /* NaN / Inf → saturate to ±6.0 */
    if (abs_a >= 0x7F800000) {
        return (sign << 3) | 0x7;
    }
    /* ±0 */
    if (abs_a == 0) {
        return sign << 3;
    }

    /*
     * Round to nearest E2M1 value using midpoints:
     *   0 ↔ 0.5  : 0.25   (0x3E800000)
     *   0.5 ↔ 1.0: 0.75   (0x3F400000)
     *   1.0 ↔ 1.5: 1.25   (0x3FA00000)
     *   1.5 ↔ 2.0: 1.75   (0x3FE00000)
     *   2.0 ↔ 3.0: 2.5    (0x40200000)
     *   3.0 ↔ 4.0: 3.5    (0x40600000)
     *   4.0 ↔ 6.0: 5.0    (0x40A00000)
     */
    uint8_t mag;
    if      (abs_a < 0x3E800000) mag = 0; /* 0   */
    else if (abs_a < 0x3F400000) mag = 1; /* 0.5 */
    else if (abs_a < 0x3FA00000) mag = 2; /* 1.0 */
    else if (abs_a < 0x3FE00000) mag = 3; /* 1.5 */
    else if (abs_a < 0x40200000) mag = 4; /* 2.0 */
    else if (abs_a < 0x40600000) mag = 5; /* 3.0 */
    else if (abs_a < 0x40A00000) mag = 6; /* 4.0 */
    else                         mag = 7; /* 6.0 */

    return (sign << 3) | mag;
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

    /* ================================================================
     * RV32F 浮点指令
     * ================================================================ */

    case OPCODE_LOAD_FP: {
        /* flw: fpr[rd] = mem[gpr[rs1] + imm_i] */
        int32_t imm = GET_IMM_I(inst);
        uint32_t addr = src1 + imm;
        lane->fpr[rd] = core_mem_read(s, addr, 4);
        break;
    }

    case OPCODE_STORE_FP: {
        /* fsw: mem[gpr[rs1] + imm_s] = fpr[rs2] */
        int32_t imm = GET_IMM_S(inst);
        uint32_t addr = src1 + imm;
        core_mem_write(s, addr, lane->fpr[rs2], 4);
        break;
    }

    case OPCODE_FMADD:
    case OPCODE_FMSUB:
    case OPCODE_FNMSUB:
    case OPCODE_FNMADD: {
        uint32_t rs3 = GET_RS3(inst);
        FloatRoundMode rm = decode_rm(funct3, lane->fcsr);
        prep_fp_status(&lane->fp_status, rm);

        int muladd_flags = 0;
        switch (opcode) {
        case OPCODE_FMSUB:
            muladd_flags = float_muladd_negate_c;
            break;
        case OPCODE_FNMSUB:
            muladd_flags = float_muladd_negate_product;
            break;
        case OPCODE_FNMADD:
            muladd_flags = float_muladd_negate_c | float_muladd_negate_product;
            break;
        }

        lane->fpr[rd] = float32_muladd(lane->fpr[rs1], lane->fpr[rs2],
                                        lane->fpr[rs3], muladd_flags,
                                        &lane->fp_status);
        sync_fp_exceptions(lane);
        break;
    }

    case OPCODE_OP_FP: {
        FloatRoundMode rm = decode_rm(funct3, lane->fcsr);
        prep_fp_status(&lane->fp_status, rm);

        switch (funct7) {
        case FUNCT7_FADD_S:
            lane->fpr[rd] = float32_add(lane->fpr[rs1], lane->fpr[rs2],
                                         &lane->fp_status);
            break;
        case FUNCT7_FSUB_S:
            lane->fpr[rd] = float32_sub(lane->fpr[rs1], lane->fpr[rs2],
                                         &lane->fp_status);
            break;
        case FUNCT7_FMUL_S:
            lane->fpr[rd] = float32_mul(lane->fpr[rs1], lane->fpr[rs2],
                                         &lane->fp_status);
            break;
        case FUNCT7_FDIV_S:
            lane->fpr[rd] = float32_div(lane->fpr[rs1], lane->fpr[rs2],
                                         &lane->fp_status);
            break;
        case FUNCT7_FSQRT_S:
            lane->fpr[rd] = float32_sqrt(lane->fpr[rs1], &lane->fp_status);
            break;
        case FUNCT7_FSGNJ_S: {
            uint32_t a = lane->fpr[rs1];
            uint32_t b = lane->fpr[rs2];
            switch (funct3) {
            case 0: /* fsgnj.s */
                lane->fpr[rd] = (a & 0x7FFFFFFF) | (b & 0x80000000);
                break;
            case 1: /* fsgnjn.s */
                lane->fpr[rd] = (a & 0x7FFFFFFF) | (~b & 0x80000000);
                break;
            case 2: /* fsgnjx.s */
                lane->fpr[rd] = a ^ (b & 0x80000000);
                break;
            default:
                goto illegal;
            }
            break;
        }
        case FUNCT7_FMINMAX_S:
            if (funct3 == 0) {
                lane->fpr[rd] = float32_minnum(lane->fpr[rs1], lane->fpr[rs2],
                                                &lane->fp_status);
            } else if (funct3 == 1) {
                lane->fpr[rd] = float32_maxnum(lane->fpr[rs1], lane->fpr[rs2],
                                                &lane->fp_status);
            } else {
                goto illegal;
            }
            break;
        case FUNCT7_FCMP_S: {
            /* feq/flt/fle.s — 结果写 gpr[rd] */
            bool result = false;
            switch (funct3) {
            case 2: /* feq.s */
                result = float32_eq_quiet(lane->fpr[rs1], lane->fpr[rs2],
                                          &lane->fp_status);
                break;
            case 1: /* flt.s */
                result = float32_lt(lane->fpr[rs1], lane->fpr[rs2],
                                    &lane->fp_status);
                break;
            case 0: /* fle.s */
                result = float32_le(lane->fpr[rs1], lane->fpr[rs2],
                                    &lane->fp_status);
                break;
            default:
                goto illegal;
            }
            lane->gpr[rd] = result ? 1 : 0;
            break;
        }
        case FUNCT7_FCVT_W_S:
            /* rs2 field selects: 0=fcvt.w.s, 1=fcvt.wu.s */
            if (rs2 == 0) {
                lane->gpr[rd] = (uint32_t)float32_to_int32(
                    lane->fpr[rs1], &lane->fp_status);
            } else {
                lane->gpr[rd] = float32_to_uint32(
                    lane->fpr[rs1], &lane->fp_status);
            }
            break;
        case FUNCT7_FCVT_S_W:
            /* rs2 field selects: 0=fcvt.s.w, 1=fcvt.s.wu */
            if (rs2 == 0) {
                lane->fpr[rd] = int32_to_float32(
                    (int32_t)lane->gpr[rs1], &lane->fp_status);
            } else {
                lane->fpr[rd] = uint32_to_float32(
                    lane->gpr[rs1], &lane->fp_status);
            }
            break;
        case FUNCT7_FMV_X_W:
            if (funct3 == 0) {
                /* fmv.x.w: gpr[rd] = fpr[rs1] (bitwise) */
                lane->gpr[rd] = lane->fpr[rs1];
            } else if (funct3 == 1) {
                /* fclass.s */
                lane->gpr[rd] = fclass_s(lane->fpr[rs1], &lane->fp_status);
            } else {
                goto illegal;
            }
            break;
        case FUNCT7_FMV_W_X:
            /* fmv.w.x: fpr[rd] = gpr[rs1] (bitwise) */
            lane->fpr[rd] = lane->gpr[rs1];
            break;

        /* ============================================================
         * Low-precision float conversions (BF16 / FP8 / FP4)
         * ============================================================ */
        case FUNCT7_FCVT_BF16:
            if (rs2 == 0) {
                /* FCVT.S.BF16: frd = bf16→f32(frs1[15:0]) */
                bfloat16 bf = (bfloat16)(lane->fpr[rs1] & 0xFFFF);
                lane->fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
            } else if (rs2 == 1) {
                /* FCVT.BF16.S: frd[15:0] = f32→bf16(frs1) */
                bfloat16 bf = float32_to_bfloat16(lane->fpr[rs1],
                                                   &lane->fp_status);
                lane->fpr[rd] = (uint32_t)bf;
            } else {
                goto illegal;
            }
            break;

        case FUNCT7_FCVT_FP8:
            switch (rs2) {
            case 0: {
                /* FCVT.S.E4M3: frd = e4m3→f32(frs1[7:0]) */
                float8_e4m3 e4 = (float8_e4m3)(lane->fpr[rs1] & 0xFF);
                bfloat16 bf = float8_e4m3_to_bfloat16(e4, &lane->fp_status);
                lane->fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
                break;
            }
            case 1: {
                /* FCVT.E4M3.S: frd[7:0] = f32→e4m3(frs1) */
                float8_e4m3 e4 = float32_to_float8_e4m3(lane->fpr[rs1],
                                                         true,
                                                         &lane->fp_status);
                lane->fpr[rd] = (uint32_t)e4;
                break;
            }
            case 2: {
                /* FCVT.S.E5M2: frd = e5m2→f32(frs1[7:0]) */
                float8_e5m2 e5 = (float8_e5m2)(lane->fpr[rs1] & 0xFF);
                bfloat16 bf = float8_e5m2_to_bfloat16(e5, &lane->fp_status);
                lane->fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
                break;
            }
            case 3: {
                /* FCVT.E5M2.S: frd[7:0] = f32→e5m2(frs1) */
                float8_e5m2 e5 = float32_to_float8_e5m2(lane->fpr[rs1],
                                                         true,
                                                         &lane->fp_status);
                lane->fpr[rd] = (uint32_t)e5;
                break;
            }
            default:
                goto illegal;
            }
            break;

        case FUNCT7_FCVT_FP4:
            if (rs2 == 0) {
                /* FCVT.S.E2M1: frd = e2m1→f32(frs1[3:0])
                 * chain: e2m1 → e4m3 → bf16 → f32 */
                float4_e2m1 e2 = (float4_e2m1)(lane->fpr[rs1] & 0xF);
                float8_e4m3 e4 = float4_e2m1_to_float8_e4m3(e2,
                                                              &lane->fp_status);
                bfloat16 bf = float8_e4m3_to_bfloat16(e4, &lane->fp_status);
                lane->fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
            } else if (rs2 == 1) {
                /* FCVT.E2M1.S: frd[3:0] = f32→e2m1(frs1)
                 * manual rounding to 4-bit E2M1 */
                float4_e2m1 e2 = float32_to_float4_e2m1(lane->fpr[rs1],
                                                          &lane->fp_status);
                lane->fpr[rd] = (uint32_t)(e2 & 0xF);
            } else {
                goto illegal;
            }
            break;

        default:
            goto illegal;
        }
        sync_fp_exceptions(lane);
        break;
    }

    case OPCODE_SYSTEM:
        /* ebreak: 停止执行 */
        if (inst == 0x00100073) {
            return 1;
        }
        /* CSR 指令 */
        if (funct3 >= FUNCT3_CSRRW && funct3 <= FUNCT3_CSRRCI) {
            uint32_t csr = GET_CSR(inst);
            uint32_t csr_val = 0;
            uint32_t write_val = 0;
            bool do_write = false;

            /* 读 CSR */
            switch (csr) {
            case CSR_MHARTID:
                csr_val = lane->mhartid;
                break;
            case CSR_FFLAGS:
                csr_val = lane->fcsr & 0x1F;
                break;
            case CSR_FRM:
                csr_val = (lane->fcsr >> 5) & 0x7;
                break;
            case CSR_FCSR:
                csr_val = lane->fcsr & 0xFF;
                break;
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "GPGPU core: unknown CSR 0x%x\n", csr);
                break;
            }

            /* 计算新 CSR 值 */
            switch (funct3) {
            case FUNCT3_CSRRW:
                write_val = src1;
                do_write = true;
                break;
            case FUNCT3_CSRRS:
                write_val = csr_val | src1;
                do_write = (rs1 != 0);
                break;
            case FUNCT3_CSRRC:
                write_val = csr_val & ~src1;
                do_write = (rs1 != 0);
                break;
            case FUNCT3_CSRRWI:
                write_val = rs1;  /* rs1 field is zimm */
                do_write = true;
                break;
            case FUNCT3_CSRRSI:
                write_val = csr_val | rs1;
                do_write = (rs1 != 0);
                break;
            case FUNCT3_CSRRCI:
                write_val = csr_val & ~rs1;
                do_write = (rs1 != 0);
                break;
            }

            /* 写入 rd */
            if (rd != 0) {
                lane->gpr[rd] = csr_val;
            }

            /* 写 CSR */
            if (do_write) {
                switch (csr) {
                case CSR_FFLAGS:
                    lane->fcsr = (lane->fcsr & ~0x1F) | (write_val & 0x1F);
                    break;
                case CSR_FRM:
                    lane->fcsr = (lane->fcsr & ~0xE0) |
                                 ((write_val & 0x7) << 5);
                    break;
                case CSR_FCSR:
                    lane->fcsr = write_val & 0xFF;
                    break;
                case CSR_MHARTID:
                    /* mhartid is read-only */
                    break;
                default:
                    break;
                }
            }
        }
        break;

    default:
    illegal:
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
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    /* 初始化每个 lane */
    for (uint32_t i = 0; i < GPGPU_WARP_SIZE; i++) {
        if (i < num_threads) {
            warp->lanes[i].pc = pc;
            warp->lanes[i].active = true;
            warp->lanes[i].mhartid =
                MHARTID_ENCODE(block_id_linear, warp_id, i);
            warp->active_mask |= (1U << i);
        } else {
            warp->lanes[i].active = false;
        }
        /* 寄存器初始化为 0 */
        memset(warp->lanes[i].gpr, 0, sizeof(warp->lanes[i].gpr));
        memset(warp->lanes[i].fpr, 0, sizeof(warp->lanes[i].fpr));
        warp->lanes[i].fcsr = 0;

        /* 初始化 softfloat 状态 */
        memset(&warp->lanes[i].fp_status, 0, sizeof(float_status));
        set_default_nan_mode(true, &warp->lanes[i].fp_status);
        set_float_default_nan_pattern(0x40, &warp->lanes[i].fp_status);
        set_float_rounding_mode(float_round_nearest_even,
                                &warp->lanes[i].fp_status);
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

            GPGPULane *lane = &warp->lanes[i];
            current_lane = lane;

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
                uint32_t block_id_linear =
                    bx + by * grid_x + bz * grid_x * grid_y;

                /* 每个 block 分成多个 warp 执行 */
                for (uint32_t tid = 0; tid < threads_per_block;
                     tid += GPGPU_WARP_SIZE) {
                    GPGPUWarp warp;
                    uint32_t num_threads = threads_per_block - tid;
                    if (num_threads > GPGPU_WARP_SIZE) {
                        num_threads = GPGPU_WARP_SIZE;
                    }

                    uint32_t warp_id = tid / GPGPU_WARP_SIZE;
                    gpgpu_core_init_warp(&warp, kernel_addr, tid, block_id,
                                         num_threads, warp_id,
                                         block_id_linear);

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
