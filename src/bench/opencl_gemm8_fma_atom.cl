#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL FP_CONTRACT OFF
#ifndef RM
#define RM 2
#endif
#ifndef RN
#define RN 2
#endif

/* The shipping scalar FMA core of libopenclmx's `gemm`, standalone and dense/plain so it can
 * be measured in the same harness as the staged kernels. Same K-block 8, same `float av[8]`
 * + `mad` chain, same accumulator order - so it is the baseline the staged kernel must beat.
 * One workgroup is a single subgroup of 8; A is read as two half4 a row and broadcast to the
 * lanes, B is gathered one half a (k, column). */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm(__global const half *A, __global const half *B,
                   __global float *C, uint M, uint N, uint K)
{
    uint lane = get_sub_group_local_id();
    uint row = get_group_id(1) * (8 * RM);
    uint col = get_group_id(0) * (8 * RN);
    size_t bt = get_group_id(2);
    A += bt * (size_t)M * K;
    B += bt * (size_t)K * N;
    C += bt * (size_t)M * N;
    float8 acc[RM][RN];
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);
    for (uint k = 0; k < K; k += 8) {
        float bv[RN][8];
#pragma unroll
        for (uint c = 0; c < RN; c++)
#pragma unroll
            for (uint j = 0; j < 8; j++)
                bv[c][j] = (float)B[(size_t)(k + j) * N + col + c * 8 + lane];
#pragma unroll
        for (uint r = 0; r < RM; r++)
#pragma unroll
            for (uint i = 0; i < 8; i++) {
                float4 lo = vload_half4(0, A + (size_t)(row + r * 8 + i) * K + k);
                float4 hi = vload_half4(0, A + (size_t)(row + r * 8 + i) * K + k + 4);
                float av[8] = { lo.s0, lo.s1, lo.s2, lo.s3, hi.s0, hi.s1, hi.s2, hi.s3 };
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
#pragma unroll
    for (uint r = 0; r < RM; r++)
#pragma unroll
        for (uint c = 0; c < RN; c++)
#pragma unroll
            for (uint i = 0; i < 8; i++)
                C[(size_t)(row + r * 8 + i) * N + col + c * 8 + lane] = acc[r][c][i];
}