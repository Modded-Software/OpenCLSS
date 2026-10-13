#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#ifndef RM
#define RM 4
#endif
#ifndef RN
#define RN 4
#endif
#ifndef KUNROLL
#define KUNROLL 1
#endif

/* opencl_gemm8.cl with block reads instead of scalar gathers (DG2, subgroup 8, 8x8x16 atom).
 *
 * B is pre-packed (weights are constant, so this is free per frame) into VNNI fragments:
 *   Bp[bt][K/16][N/8][i][lane] = (B[k0 + 2i][n0 + lane], B[k0 + 2i + 1][n0 + lane])
 * so one atom's B operand is 64 contiguous uints - a single intel_sub_group_block_read8,
 * where opencl_gemm8.cl spent 16 scalar 16-bit loads.  A stays row-major: row r of an atom
 * is 8 lanes x 32 bits contiguous, one intel_sub_group_block_read per row.
 */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm(__global const half *A, __global const uint *Bp,
                   __global float *C, uint M, uint N, uint K)
{
    uint lane = get_sub_group_local_id();
    uint row = get_group_id(1) * (8 * RM);
    uint col = get_group_id(0) * (8 * RN);
    size_t bt = get_group_id(2);
    __global const uint *A32 = (__global const uint *)(A + bt * (size_t)M * K);
    Bp += bt * (size_t)K * N / 2;
    C += bt * (size_t)M * N;
    uint ntiles = N / 8;
    float8 acc[RM][RN];
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);
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
                    a[u][r][i] = as_int(intel_sub_group_block_read(
                        A32 + ((size_t)(row + r * 8 + i) * K + kk) / 2));
#pragma unroll
            for (uint c = 0; c < RN; c++)
                b[u][c] = as_int8(intel_sub_group_block_read8(
                    Bp + ((size_t)(kk / 16) * ntiles + col / 8 + c) * 64));
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
