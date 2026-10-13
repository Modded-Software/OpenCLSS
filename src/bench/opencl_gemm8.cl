#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#ifndef RM
#define RM 1
#endif
#ifndef RN
#define RN 1
#endif
#ifndef KUNROLL
#define KUNROLL 1
#endif

/* Intel's min-subgroup-size-8 (Xe-HPG / DG2 / Arc-A) DPAS interface.
 *
 * The device's minimum subgroup size is 8, so the subgroup MUST be 8 and the atom is
 * 8x8x16 (M x N x K).  From cl_intel_subgroup_matrix_multiply_accumulate:
 *
 *   float8 intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc);  // M=8
 *
 *   a : M=8 rows x K=16 columns.  Each work item contributes 32 bits (2 fp16) per row,
 *       so 8 lanes x 2 = 16 = K.  int8 a => a[r] packs row r's (k=2l, k=2l+1).
 *   b : K=16 rows x N=8 columns.  Each work item contributes one column (lane), 16 K
 *       values packed into int8 => b[i] packs (k=2i, k=2i+1) of column `lane`.
 *   acc: M=8 x N=1 (this lane's column), fp32.
 *
 * RM row-blocks x RN column-blocks of the 8x8 atom = register blocking.  KUNROLL batches
 * the global loads of several K steps ahead of their mads, so the DPAS has independent
 * operands to issue against while the next loads are in flight (the short-K shapes here
 * are latency-bound, not unit-bound: 8x8 one atom is 7.4 TFLOP/s, four atoms 14.5).
 */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm(__global const half *A, __global const half *B,
                   __global float *C, uint M, uint N, uint K)
{
    uint lane = get_sub_group_local_id();
    uint row = get_group_id(1) * (8 * RM);
    uint col = get_group_id(0) * (8 * RN);
    /* One dispatch serves all `batch` GEMMs of the shape: extra grid dimension, per-batch
       base pointers.  A 2D range leaves get_group_id(2)==0 so the plain path is unchanged. */
    size_t bt = get_group_id(2);
    A += bt * (size_t)M * K;
    B += bt * (size_t)K * N;
    C += bt * (size_t)M * N;
float8 acc[RM][RN];
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);
#ifdef VEC
    __global const half2 *A2 = (__global const half2 *)A;
#endif
    for (uint k = 0; k < K; k += 16 * KUNROLL) {
        int8 a[KUNROLL][RM];
        int8 b[KUNROLL][RN];
#pragma unroll
        for (uint u = 0; u < KUNROLL; u++) {
            uint kk = k + 16 * u;
#pragma unroll
            for (uint r = 0; r < RM; r++)
#pragma unroll
                for (uint i = 0; i < 8; i++)
#ifdef VEC
                    a[u][r][i] = as_int(A2[((row + r * 8 + i) * K + kk) / 2 + lane]);
#else
                    a[u][r][i] = as_int((half2)(A[(row + r * 8 + i) * K + kk + 2 * lane],
                                                A[(row + r * 8 + i) * K + kk + 2 * lane + 1]));
#endif
#pragma unroll
            for (uint c = 0; c < RN; c++)
#pragma unroll
                for (uint i = 0; i < 8; i++)
                    b[u][c][i] = as_int((half2)(B[(kk + 2 * i) * N + col + c * 8 + lane],
                                                B[(kk + 2 * i + 1) * N + col + c * 8 + lane]));
        }
#pragma unroll
        for (uint u = 0; u < KUNROLL; u++)
#pragma unroll
            for (uint r = 0; r < RM; r++)
#pragma unroll
                for (uint c = 0; c < RN; c++)
                    acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a[u][r], b[u][c], acc[r][c]);
    }
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++)
#pragma unroll
            for (uint i = 0; i < 8; i++)
                C[(row + r * 8 + i) * N + col + c * 8 + lane] = acc[r][c][i];
}