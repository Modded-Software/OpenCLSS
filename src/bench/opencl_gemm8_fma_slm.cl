#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL FP_CONTRACT OFF
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 32
#endif
#ifndef RM
#define RM 2
#endif
#ifndef RN
#define RN 2
#endif
#define SGM (BM / (8 * RM))
#define SGN (BN / (8 * RN))
#define WGS (8 * SGM * SGN)

/* Deterministic SLM-staged GEMM: the register-blocked FMA core of libopenclmx's `gemm`,
 * with A and B staged into shared local memory once per workgroup instead of each subgroup
 * gathering them from global.
 *
 * The scalar kernel's cost is not the multiply-adds: its IGC disassembly is dominated by
 * address arithmetic and register moves, and every subgroup re-issues the same global
 * loads. Here the workgroup's WGS threads stage a BM x BK slab of A and a BK x BN slab of
 * B once (coalesced), and every subgroup reads its fragments from SLM. B is shared down the
 * workgroup's rows and A across its columns, so the global traffic per multiply falls by
 * SGM and SGN.
 *
 * The FMA core is bit-identical to libopenclmx's: `acc += a0*b0 + ... + a7*b7` with
 * FP_CONTRACT OFF, ordered as a `mad` chain per accumulator. No DPAS intrinsic, so no IGC
 * SWSB hazard. A is read [row][k], B [k][col]. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm(__global const half *A, __global const half *B,
                   __global float *C, uint M, uint N, uint K)
{
    __local half sA[BM][BK];
    __local half sB[BK][BN];
    uint lid = get_local_id(0);
    uint lane = get_sub_group_local_id();
    uint sg = lid / 8;
    uint sg_m = sg / SGN;
    uint sg_n = sg % SGN;
    uint blockRow = get_group_id(1) * BM;
    uint blockCol = get_group_id(0) * BN;
    uint myRow = sg_m * (8 * RM);
    uint myCol = sg_n * (8 * RN);
    float8 acc[RM][RN];
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (uint idx = lid; idx < BM * BK; idx += WGS) {
            uint row = idx / BK, col = idx % BK;
            sA[row][col] = A[(blockRow + row) * (size_t)K + k0 + col];
        }
#pragma unroll
        for (uint idx = lid; idx < BK * BN; idx += WGS) {
            uint k = idx / BN, col = idx % BN;
            sB[k][col] = B[(k0 + k) * (size_t)N + blockCol + col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 0; kk < BK; kk += 8) {
            float bv[RN][8];
#pragma unroll
            for (uint c = 0; c < RN; c++)
#pragma unroll
                for (uint j = 0; j < 8; j++)
                    bv[c][j] = (float)sB[kk + j][myCol + c * 8 + lane];
#pragma unroll
            for (uint r = 0; r < RM; r++)
#pragma unroll
                for (uint i = 0; i < 8; i++) {
                    float av[8];
#pragma unroll
                    for (uint j = 0; j < 8; j++)
                        av[j] = (float)sA[myRow + r * 8 + i][kk + j];
#pragma unroll
                    for (uint c = 0; c < RN; c++) {
                        float s = acc[r][c][i];
#pragma unroll
                        for (uint j = 0; j < 8; j++)
                            s = mad(av[j], bv[c][j], s);
                        acc[r][c][i] = s;
                    }
                }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    uint col = blockCol + myCol;
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++)
#pragma unroll
            for (uint i = 0; i < 8; i++)
                C[(size_t)(blockRow + myRow + r * 8 + i) * N + col + c * 8 + lane] = acc[r][c][i];
}