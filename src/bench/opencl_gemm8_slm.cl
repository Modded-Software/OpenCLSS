#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
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

/* DG2 subgroup-8 DPAS with shared-memory staging.
 *
 * The atom kernel (opencl_gemm8.cl) is load-instruction bound: every DPAS is fed by ~16
 * scalar global loads.  Here a whole workgroup stages a BM x BK slab of A and a BK x BN
 * slab of B into SLM once, and every subgroup then reads its fragments from SLM.  B is
 * shared down the workgroup's columns and A across its rows, so the global traffic per
 * multiply falls by SGM and SGN respectively, and the fragment loads become local.
 *
 * B is staged transposed (BN x BK) so that a lane's (k=2i, 2i+1) pair is one aligned
 * half2 - the same trick as -DVEC for A.
 */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm(__global const half *A, __global const half *B,
                   __global float *C, uint M, uint N, uint K)
{
    __local half sA[BM][BK];
    __local half sBt[BN][BK];
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
    for (uint r = 0; r < RM; r++)
        for (uint c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint idx = lid; idx < BM * BK; idx += WGS) {
            uint row = idx / BK, col = idx % BK;
            sA[row][col] = A[(blockRow + row) * K + k0 + col];
        }
        for (uint idx = lid; idx < BN * BK; idx += WGS) {
            uint n = idx / BK, k = idx % BK;
            sBt[n][k] = B[(k0 + k) * N + blockCol + n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 0; kk < BK; kk += 16) {
            int8 a[RM], b[RN];
            for (uint r = 0; r < RM; r++)
                for (uint i = 0; i < 8; i++)
                    a[r][i] = as_int(*(__local half2 *)&sA[myRow + r * 8 + i][kk + 2 * lane]);
            for (uint c = 0; c < RN; c++)
                for (uint i = 0; i < 8; i++)
                    b[c][i] = as_int(*(__local half2 *)&sBt[myCol + c * 8 + lane][kk + 2 * i]);
            for (uint r = 0; r < RM; r++)
                for (uint c = 0; c < RN; c++)
                    acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a[r], b[c], acc[r][c]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    uint col = blockCol + myCol;
    for (uint r = 0; r < RM; r++)
        for (uint c = 0; c < RN; c++)
            for (uint i = 0; i < 8; i++)
                C[(blockRow + myRow + r * 8 + i) * N + col + c * 8 + lane] = acc[r][c][i];
}