/*
 * libopenclmx - the xmx_* runtime on OpenCL, for a device whose matrix unit is SIMD8.
 *
 * Why this exists: VK_KHR_cooperative_matrix pins the subgroup to 32 on this part, while
 * the XMX DPAS unit is SIMD8 (notes/phase26-the-register-ceiling.md).  An 8x16 fp32
 * accumulator is then four registers a lane and the register file caps the tile.  OpenCL
 * can declare the subgroup the hardware actually has - 8 - and feed the DPAS intrinsic
 * directly (cl_intel_subgroup_matrix_multiply_accumulate), which is what this backend is
 * for.  It is the same xmx_* entry points as libxmx; NR_GPU_BACKEND=opencl selects it.
 *
 * Buffers are host-addressable: the whole graph's buffers come from malloc and are handed
 * to OpenCL with CL_MEM_USE_HOST_PTR, so xmx_buf_ptr returns the same memory the host
 * wrote and there is no staging copy.  Command recording is a list of launches replayed
 * in order at submit; there are no Vulkan command buffers here.
 *
 * This is the first slice: device, buffers, recording, copies, the plain GEMM and the
 * device queries.  The fused passes and the elementwise families fall through to
 * "not implemented" and are added one at a time, each against its Vulkan test.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

/* ---- the OpenCL entry points this backend uses, declared by hand ------------------- */
/* No CL/cl.h on this box; the runtime is libOpenCL.so.1 with no unversioned symlink, so
 * the build links the versioned name (the Makefile rule passes it). */
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_bitfield, cl_device_type, cl_command_queue_properties, cl_ulong;
typedef int64_t cl_long;
typedef size_t cl_size_t;
typedef void *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

#define CL_SUCCESS 0
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_TRUE 1
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)

extern cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
extern cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
extern cl_context clCreateContext(const cl_int *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
extern cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
extern cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
extern cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
extern cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
extern cl_kernel clCreateKernel(cl_program, const char *, cl_int *);
extern cl_mem clCreateBuffer(cl_context, cl_bitfield, size_t, void *, cl_int *);
extern cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);
extern cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *);
extern cl_int clEnqueueCopyBuffer(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const void *, void *);
extern cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_int, size_t, size_t, void *, cl_uint, const void *, void *);
extern cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_int, size_t, size_t, const void *, cl_uint, const void *, void *);
extern cl_int clEnqueueFillBuffer(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const void *, void *);
extern cl_int clReleaseMemObject(cl_mem);
extern cl_int clFinish(cl_command_queue);
extern cl_int clReleaseCommandQueue(cl_command_queue);
extern cl_int clFlush(cl_command_queue);
extern cl_int clGetEventProfilingInfo(cl_event, cl_uint, size_t, void *, size_t *);
extern cl_int clReleaseEvent(cl_event);
extern void *clEnqueueMapBuffer(cl_command_queue, cl_mem, cl_int, cl_bitfield, size_t, size_t, cl_uint, const void *, void *, cl_int *);
extern cl_int clEnqueueUnmapMemObject(cl_command_queue, cl_mem, void *, cl_uint, const void *, void *);

/* ---- the SIMD8 DPAS GEMM kernel --------------------------------------------------- */
/* One subgroup of 8 per workgroup; RM x RN 8x8x16 atoms.  Batch is grid dimension 2 with
 * per-batch base pointers, so one dispatch serves a whole batched GEMM as the frame asks. */
static const char *GEMM_SRC =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
"#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
"#ifndef RM\n#define RM 2\n#endif\n"
"#ifndef RN\n#define RN 2\n#endif\n"
"#ifdef GEMM_DPAS\n"
"/* A's 8 rows and B's 8 columns as the DPAS k16 operands: 16 halves packed two to an int,\n"
" * filled by integer ops on the raw ushort words. */\n"
"static inline int8 xmx_dpas_a(__global const ushort *Au, size_t row0, size_t lda,\n"
"                              unsigned k, unsigned lane) {\n"
"    int8 t;\n"
"#pragma unroll\n"
"    for (int i = 0; i < 8; i++) {\n"
"        size_t at = (row0 + (size_t)i) * lda + k + 2 * lane;\n"
"        t[i] = (int)((uint)Au[at] | ((uint)Au[at + 1] << 16));\n"
"    }\n"
"    return t;\n"
"}\n"
"static inline int8 xmx_dpas_b(__global const ushort *Bu, size_t col0, size_t ldb,\n"
"                              unsigned k, unsigned lane, unsigned transposed) {\n"
"    int8 t;\n"
"#pragma unroll\n"
"    for (int i = 0; i < 8; i++) {\n"
"        uint lo, hi;\n"
"        if (transposed) {\n"
"            size_t at = (col0 + (size_t)lane) * ldb + k + 2 * i;\n"
"            lo = Bu[at]; hi = Bu[at + 1];\n"
"        } else {\n"
"            lo = Bu[(size_t)(k + 2 * i) * ldb + col0 + (size_t)lane];\n"
"            hi = Bu[(size_t)(k + 2 * i + 1) * ldb + col0 + (size_t)lane];\n"
"        }\n"
"        t[i] = (int)(lo | (hi << 16));\n"
"    }\n"
"    return t;\n"
"}\n"
"#endif\n"
/* Row strides `ld`, element offsets `o` and batch strides `s` are the frame's own: a GEMM
 * reads or writes a slice of a wider buffer and batches it. Host resolves the zero-means-
 * dense defaults, so this kernel always sees real strides. `transposed` is B stored [N][K]. */
"__attribute__((intel_reqd_sub_group_size(8)))\n"
"__kernel void gemm(__global const half *A, __global const half *B, __global float *C,\n"
"                   __global const half *Sk, __global const float *Sf,\n"
"                   __global const float *Cos,\n"
"                   unsigned M, unsigned N, unsigned K,\n"
"                   unsigned lda, unsigned ldb, unsigned ldc,\n"
"                   unsigned oa, unsigned ob, unsigned oc,\n"
"                   unsigned sa, unsigned sb, unsigned sc, unsigned transposed,\n"
"                   unsigned epilogue, unsigned narrow, unsigned compact,\n"
"                   unsigned residual, unsigned skip_half,\n"
"                   unsigned window_res, unsigned win_h, unsigned win_w,\n"
"                   unsigned win_across, unsigned win_pad)\n"
"{\n"
"    unsigned lane = get_sub_group_local_id();\n"
"    __global half *Ch = (__global half *)C;\n"
"    unsigned row = get_group_id(1) * (8 * RM);\n"
"    unsigned col = get_group_id(0) * (8 * RN);\n"
"    size_t bt = get_group_id(2);\n"
"    A += oa + bt * (size_t)sa; B += ob + bt * (size_t)sb;\n" \
"    /* C's offsets are in output elements: move the float and half views apart, each in\n" \
"     * its own width, since only one of them is written. */\n" \
"    C += oc + bt * (size_t)sc; Ch += oc + bt * (size_t)sc;\n"
"    float8 acc[RM][RN];\n"
"#pragma unroll\n"
"    for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++)\n"
"            acc[r][c] = (float8)(0.0f);\n"
"#ifndef GEMM_DPAS\n"
"    /* A plain half->float FMA core, no DPAS intrinsic - the default. The matrix intrinsic\n"
"     * is correct in the ISA but the optimiser miscompiles one accumulator atom of it on\n"
"     * this IGC, run to run, on the frame's batched QK shapes; -cl-opt-disable hides it at\n"
"     * 6.6x the time, and this core is the deterministic path. -DGEMM_DPAS selects the\n"
"     * intrinsic (notes/AUDIT-opencl-defects.md). k is stepped two at a time and A read as\n"
"     * half2: the kernel is load-bound, and this halves the A load instructions. */\n"
"    /* Register-blocked FMA, K blocked 8: B's 8 K-values per column atom are held once, then\n"
"     * each A row is loaded (two half4) and reused across all RN columns. Materialising all\n"
"     * RM*8 A rows up front costs RM*8*8 live floats and caps occupancy; this keeps only B\n"
"     * (RN*8), one A row (8) and the accumulators live. */\n"
"    for (unsigned k = 0; k < K; k += 8) {\n"
"        float bv[RN][8];\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++) {\n"
"            unsigned bcol = col + c * 8 + lane;\n"
"            if (transposed) {\n"
"                float4 lo = vload_half4(0, B + (size_t)bcol * ldb + k);\n"
"                float4 hi = vload_half4(0, B + (size_t)bcol * ldb + k + 4);\n"
"                bv[c][0] = lo.s0; bv[c][1] = lo.s1; bv[c][2] = lo.s2; bv[c][3] = lo.s3;\n"
"                bv[c][4] = hi.s0; bv[c][5] = hi.s1; bv[c][6] = hi.s2; bv[c][7] = hi.s3;\n"
"            } else {\n"
"#pragma unroll\n"
"                for (unsigned j = 0; j < 8; j++)\n"
"                    bv[c][j] = (float)B[(size_t)(k + j) * ldb + bcol];\n"
"            }\n"
"        }\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"            for (unsigned i = 0; i < 8; i++) {\n"
"                float4 lo = vload_half4(0, A + (size_t)(row + r * 8 + i) * lda + k);\n"
"                float4 hi = vload_half4(0, A + (size_t)(row + r * 8 + i) * lda + k + 4);\n"
"                float av[8] = { lo.s0, lo.s1, lo.s2, lo.s3,\n"
"                                hi.s0, hi.s1, hi.s2, hi.s3 };\n"
"                /* The accumulation must not be contracted into an fma by the compiler:\n"
"                 * FP_CONTRACT OFF is global because the publish epilogue's rounding is the\n"
"                 * vendor's, but that also splits this sum into mul+add, halving the FP\n"
"                 * rate (notes/AUDIT-opencl-defects.md). So the fma is asked for explicitly,\n"
"                 * with `mad`, which stays fused under FP_CONTRACT OFF and gives the same\n"
"                 * answer every run - the epilogue's rounding is untouched. */\n"
"#pragma unroll\n"
"                for (unsigned c = 0; c < RN; c++) {\n"
"                    float s = acc[r][c][i];\n"
"#pragma unroll\n"
"                    for (unsigned j = 0; j < 8; j++)\n"
"                        s = mad(av[j], bv[c][j], s);\n"
"                    acc[r][c][i] = s;\n"
"                }\n"
"            }\n"
"    }\n"
"#else\n"
"    __global const ushort *Au = (__global const ushort *)A, *Bu = (__global const ushort *)B;\n"
"    /* Two independent operand sets a step (K unrolled by 2): a DPAS must not share operand\n"
"     * registers with the loads that follow it across the loop back-edge - that WAR is what\n"
"     * IGC's SWSB pass gets wrong here (notes/AUDIT-opencl-defects.md). */\n"
"    unsigned k = 0;\n"
"    for (; k + 32 <= K; k += 32) {\n"
"        int8 a0[RM], b0[RN], a1[RM], b1[RN];\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++) {\n"
"            a0[r] = xmx_dpas_a(Au, row + r * 8, lda, k, lane);\n"
"            a1[r] = xmx_dpas_a(Au, row + r * 8, lda, k + 16, lane);\n"
"        }\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++) {\n"
"            b0[c] = xmx_dpas_b(Bu, col + c * 8, ldb, k, lane, transposed);\n"
"            b1[c] = xmx_dpas_b(Bu, col + c * 8, ldb, k + 16, lane, transposed);\n"
"        }\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"            for (unsigned c = 0; c < RN; c++) {\n"
"                acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a0[r], b0[c], acc[r][c]);\n"
"                acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a1[r], b1[c], acc[r][c]);\n"
"            }\n"
"    }\n"
"    if (k < K) {\n"
"        int8 a0[RM], b0[RN];\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++)\n"
"            a0[r] = xmx_dpas_a(Au, row + r * 8, lda, k, lane);\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++)\n"
"            b0[c] = xmx_dpas_b(Bu, col + c * 8, ldb, k, lane, transposed);\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"            for (unsigned c = 0; c < RN; c++)\n"
"                acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a0[r], b0[c], acc[r][c]);\n"
"    }\n"
"#endif\n"
"#pragma unroll\n"
"    for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++)\n"
"#pragma unroll\n"
"            for (unsigned i = 0; i < 8; i++) {\n"
"                unsigned row_i = row + r * 8 + i;\n"
"                unsigned cos_i = col + c * 8 + lane;\n"
"                unsigned at = row_i * ldc + cos_i;\n"
"                int ok = 1;\n"
"                if (window_res) {\n"
"                    unsigned token = row_i & 63u, window = row_i >> 6;\n"
"                    unsigned y = (window / win_across) * 8u + (token >> 3);\n"
"                    unsigned x = (window % win_across) * 8u + (token & 7u);\n"
"                    unsigned top = win_pad >> 16, left = win_pad & 0xffffu;\n"
"                    if (y < top || x < left || y - top >= win_h || x - left >= win_w)\n"
"                        ok = 0;\n"
"                    else at = ((y - top) * win_w + (x - left)) * N + cos_i;\n"
"                }\n"
"                float v = acc[r][c][i];\n"
"                if (residual && ok) v += (skip_half ? (float)Sk[at] : Sf[at]) * Cos[cos_i];\n"
"                v = publish_epi(epilogue, v);\n"
"                if (ok) {\n"
"                    if (compact) { if (cos_i < 4) C[at] = v; }\n"
"                    else if (narrow) Ch[at] = (half)v;\n"
"                    else C[at] = v;\n"
"                }\n"
"            }\n"
"}\n";

/* ---- the block-read GEMM and its B packing (P1, from src/bench/opencl_gemm8_blk.cl) --- */
/* OFF by default (XMX_CL_BLK=1 turns it on). Wired in and passing the isolated GEMM,
 * residual, epilogue and determinism tests, but in the frame it is not yet exact: with the
 * epilogues enabled test_input_fp16.py diverges between the single- and block-schedules, and
 * even restricted to plain GEMMs (bt==0) it makes test_compact_head.py fail about one run in
 * three where the scalar kernel is 3/3 - the same IGC DPAS nondeterminism the RM=2 scalar
 * build shows, now in the block-read form. The RM=1 scalar kernel stays the default until
 * this is exact; see the diary, Run 30. */
/* B is pre-packed into the VNNI fragments the DPAS B operand wants, so one
 * intel_sub_group_block_read8 fetches a whole 16x8 atom where the scalar kernel spent 16
 * gathers. A stays row-major and is read one intel_sub_group_block_read a row. Fragments
 * are one uint each: even k in the low half, so lane l holds (B[k0+2i][n0+l],
 * B[k0+2i+1][n0+l]). */
static const char *PACK_SRC =
"__attribute__((intel_reqd_sub_group_size(8)))\n"
"__kernel void pack(__global const ushort *B, __global uint *Bp,\n"
"                   unsigned K, unsigned N, unsigned ldb, unsigned sb)\n"
"{\n"
"    unsigned gtile = get_global_id(0);\n"
"    unsigned ntiles = N / 8u, ktiles = K / 16u;\n"
"    unsigned bt = gtile / (ktiles * ntiles), rem = gtile % (ktiles * ntiles);\n"
"    unsigned kt = rem / ntiles, nt = rem % ntiles;\n"
"    __global const ushort *Bbt = B + (size_t)bt * sb;\n"
"    __global uint *out = Bp + ((size_t)bt * ktiles * ntiles + (size_t)kt * ntiles + nt) * 64u;\n"
"    unsigned k0 = kt * 16u, n0 = nt * 8u;\n"
"    for (unsigned i = 0; i < 8; i++)\n"
"        for (unsigned l = 0; l < 8; l++) {\n"
"            unsigned lo = Bbt[(size_t)(k0 + 2 * i) * ldb + n0 + l];\n"
"            unsigned hi = Bbt[(size_t)(k0 + 2 * i + 1) * ldb + n0 + l];\n"
"            out[i * 8u + l] = lo | (hi << 16);\n"
"        }\n"
"}\n";

/* The same rows, epilogues and strides as GEMM_SRC, but A and the packed B read by block.
 * Routing (run_gemm) only sends a dense, untransposed B here; A's row stride must be even,
 * which dense K (a multiple of 16) makes so. RM x RN atoms a workgroup. */
static const char *GEMM_BLK_SRC =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
"#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
"#ifndef RM\n#define RM 1\n#endif\n"
"#ifndef RN\n#define RN 1\n#endif\n"
"__attribute__((intel_reqd_sub_group_size(8)))\n"
"__kernel void gemm(__global const half *A, __global const uint *Bp, __global float *C,\n"
"                   __global const half *Sk, __global const float *Sf,\n"
"                   __global const float *Cos,\n"
"                   unsigned M, unsigned N, unsigned K,\n"
"                   unsigned lda, unsigned ldb, unsigned ldc,\n"
"                   unsigned oa, unsigned ob, unsigned oc,\n"
"                   unsigned sa, unsigned sb, unsigned sc, unsigned transposed,\n"
"                   unsigned epilogue, unsigned narrow, unsigned compact,\n"
"                   unsigned residual, unsigned skip_half,\n"
"                   unsigned window_res, unsigned win_h, unsigned win_w,\n"
"                   unsigned win_across, unsigned win_pad)\n"
"{\n"
"    unsigned lane = get_sub_group_local_id();\n"
"    __global half *Ch = (__global half *)C;\n"
"    unsigned row = get_group_id(1) * (8 * RM);\n"
"    unsigned col = get_group_id(0) * (8 * RN);\n"
"    size_t bt = get_group_id(2);\n"
"    A += oa + bt * (size_t)sa;\n"
"    __global const uint *A32 = (__global const uint *)A;\n"
"    unsigned ntiles = N / 8u;\n"
"    Bp += bt * (size_t)(K * N / 2u);\n"
"    C += oc + bt * (size_t)sc; Ch += oc + bt * (size_t)sc;\n"
"    float8 acc[RM][RN];\n"
"#pragma unroll\n"
"    for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);\n"
"    for (unsigned k = 0; k < K; k += 16) {\n"
"        int8 a[RM], b[RN];\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"            for (unsigned i = 0; i < 8; i++)\n"
"                a[r][i] = as_int(intel_sub_group_block_read(\n"
"                    A32 + ((size_t)(row + r * 8 + i) * lda + k) / 2));\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++)\n"
"            b[c] = as_int8(intel_sub_group_block_read8(\n"
"                Bp + ((size_t)(k / 16) * ntiles + col / 8 + c) * 64));\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"            for (unsigned c = 0; c < RN; c++)\n"
"                acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a[r], b[c], acc[r][c]);\n"
"    }\n"
"#pragma unroll\n"
"    for (unsigned r = 0; r < RM; r++)\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++)\n"
"            for (unsigned i = 0; i < 8; i++) {\n"
"                unsigned row_i = row + r * 8 + i;\n"
"                unsigned cos_i = col + c * 8 + lane;\n"
"                unsigned at = row_i * ldc + cos_i;\n"
"                int ok = 1;\n"
"                if (window_res) {\n"
"                    unsigned token = row_i & 63u, window = row_i >> 6;\n"
"                    unsigned y = (window / win_across) * 8u + (token >> 3);\n"
"                    unsigned x = (window % win_across) * 8u + (token & 7u);\n"
"                    unsigned top = win_pad >> 16, left = win_pad & 0xffffu;\n"
"                    if (y < top || x < left || y - top >= win_h || x - left >= win_w)\n"
"                        ok = 0;\n"
"                    else at = ((y - top) * win_w + (x - left)) * N + cos_i;\n"
"                }\n"
"                float v = acc[r][c][i];\n"
"                if (residual && ok) v += (skip_half ? (float)Sk[at] : Sf[at]) * Cos[cos_i];\n"
"                v = publish_epi(epilogue, v);\n"
"                if (ok) {\n"
"                    if (compact) { if (cos_i < 4) C[at] = v; }\n"
"                    else if (narrow) Ch[at] = (half)v;\n"
"                    else C[at] = v;\n"
"                }\n"
"            }\n"
"}\n";

/* ---- the vendor's rounding points, shared by every kernel (publish.glsl) ----------- */
/* `half_round_bits` is done on the float's bits rather than `(half)x`: Intel's and
 * Mesa's compilers disagree over whether the cast or the round trip survives, and only
 * the bit form is the same everywhere (publish.glsl, phase71). FP_CONTRACT OFF keeps the
 * gate's multiply and add from becoming one fma, which is the vendor's rounding point. */
#define CL_COMMON \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#pragma OPENCL FP_CONTRACT OFF\n" \
"inline float half_round_bits(float x) {\n" \
"    uint bits = as_uint(x), magnitude = bits & 0x7FFFFFFFu, sign = bits & 0x80000000u;\n" \
"    if (magnitude > 0x7F800000u) return x;\n" \
"    if (magnitude >= 0x477FF000u) return as_float(sign | 0x7F800000u);\n" \
"    if (magnitude >= 0x38800000u)\n" \
"        return as_float(sign | ((magnitude + 0x0FFFu + ((magnitude >> 13) & 1u)) & ~0x1FFFu));\n" \
"    uint exponent = magnitude >> 23;\n" \
"    if (exponent < 101u) return as_float(sign);\n" \
"    uint mantissa = (magnitude & 0x7FFFFFu) | 0x800000u, drop = 126u - exponent;\n" \
"    uint steps = (mantissa + ((1u << (drop - 1u)) - 1u) + ((mantissa >> drop) & 1u)) >> drop;\n" \
"    return as_float(sign | as_uint((float)steps * 5.9604644775390625e-08f));\n" \
"}\n" \
"#define HROUND(x) half_round_bits(x)\n" \
"inline float e4m3(float x) {\n" \
"    float magnitude = fmin(fabs(x), 448.0f);\n" \
"    int exponent = (as_int(magnitude) >> 23) & 0xFF;\n" \
"    exponent = max(exponent, 121) - 3;\n" \
"    float step = as_float(exponent << 23);\n" \
"    float reciprocal = as_float((254 - exponent) << 23);\n" \
"    float rounded = rint(magnitude * reciprocal) * step;\n" \
"    return x < 0.0f ? -rounded : rounded;\n" \
"}\n" \
"inline float gate_activation(float x) {\n" \
"    float wide = HROUND(x);\n" \
"    float clamped = clamp(wide, -4.0f, 4.0f);\n" \
"    float linear = fabs(clamped) * -0.055908203125f;\n" \
"    linear += 0.447265625f;\n" \
"    linear = HROUND(linear);\n" \
"    linear *= clamped;\n" \
"    linear += 0.89453125f;\n" \
"    linear = HROUND(linear);\n" \
"    return HROUND(wide * linear);\n" \
"}\n" \
"inline float publish_epi(uint epilogue, float value) {\n" \
"    if (epilogue == 2u || epilogue == 3u) value = gate_activation(value);\n" \
"    if (epilogue == 1u || epilogue == 3u) value = e4m3(value);\n" \
"    if (epilogue == 4u) value = HROUND(value);\n" \
"    return value;\n" \
"}\n"

/* ---- resident.comp - the elementwise half of the graph ---------------------------- */
#define RESIDENT_SRC CL_COMMON \
"#define E4M3 0u\n#define GATE 1u\n#define HALF 2u\n#define TO_HALF 3u\n#define SCALE 4u\n" \
"#define RESIDUAL 5u\n#define FROM_HALF 6u\n#define PARTITION 7u\n#define REVERSE 8u\n" \
"#define ADD_BIAS 9u\n#define SPLIT_HEADS 10u\n#define MERGE_HEADS 11u\n#define POOL2 12u\n" \
"#define UPSAMPLE2 13u\n#define SCALE_CHANNEL 14u\n#define ADD 15u\n#define PAD_END 16u\n" \
"#define GATE_E4M3_HALF 17u\n#define E4M3_HALF 18u\n#define GATE_HALF 19u\n" \
"#define UPSAMPLE_MERGE 20u\n#define UPSAMPLE_ADD 21u\n#define POOL2_SKIP 22u\n" \
"__kernel void resident(uint flags, __global const float *A, __global const float *B,\n" \
"                       __global float *C, __global const float *D, __global float *S,\n" \
"                       uint m, uint n, uint k, uint batch, uint sa, uint sb, uint sc, float p0)\n" \
"{\n" \
"    uint index = get_global_id(0);\n" \
"    if (index >= m) return;\n" \
"    uint kind = flags & 0xFFu, epilogue = (flags >> 8) & 0xFu;\n" \
"    int ahalf = (flags & 0x8000u) != 0u, bhalf = (flags & 0x10000u) != 0u;\n" \
"    __global const half *Ah = (__global const half *)A, *Bh = (__global const half *)B;\n" \
"    __global half *Ch = (__global half *)C, *Sh = (__global half *)S;\n" \
"    float value;\n" \
"    uint c, rest;\n" \
"    switch (kind) {\n" \
"    case GATE_E4M3_HALF: case E4M3_HALF: case GATE_HALF:\n" \
"        value = A[index];\n" \
"        if (kind != E4M3_HALF) value = gate_activation(value);\n" \
"        if (kind != GATE_HALF) value = e4m3(value);\n" \
"        Ch[index] = (half)value; return;\n" \
"    case TO_HALF: Ch[index] = (half)(A[index] * p0); return;\n" \
"    case FROM_HALF:\n" \
"        value = (float)Ah[index] * p0;\n" \
"        goto store;\n" \
"    case RESIDUAL: {\n" \
"        uint at = index;\n" \
"        if ((flags & 0x4000u) != 0u) {\n" \
"            uint width = sb, size = batch, across = sc;\n" \
"            c = index % n; rest = index / n;\n" \
"            uint py = rest / width + (k >> 16), px = rest % width + (k & 0xFFFFu);\n" \
"            at = (((py / size) * across + px / size) * size * size\n" \
"                  + (py % size) * size + px % size) * n + c;\n" \
"        }\n" \
"        value = (ahalf ? (float)Ah[at] : A[at]) + (bhalf ? (float)Bh[index] : B[index]) * D[index % n];\n" \
"        goto store;\n" \
"    }\n" \
"    case PARTITION: case REVERSE: {\n" \
"        uint size = batch, height = sa, width = sb, across = sc;\n" \
"        uint pad_top = k >> 16, pad_left = k & 0xFFFFu;\n" \
"        c = index % n; rest = index / n;\n" \
"        if (kind == PARTITION) {\n" \
"            uint token = rest % (size * size), window = rest / (size * size);\n" \
"            int y = (int)((window / across) * size + token / size) - (int)pad_top;\n" \
"            int x = (int)((window % across) * size + token % size) - (int)pad_left;\n" \
"            value = (y < 0 || x < 0 || y >= (int)height || x >= (int)width) ? 0.0f\n" \
"                  : (ahalf ? (float)Ah[((uint)y * width + (uint)x) * n + c]\n" \
"                           : A[((uint)y * width + (uint)x) * n + c]);\n" \
"        } else {\n" \
"            uint x = rest % width, y = rest / width;\n" \
"            uint py = y + pad_top, px = x + pad_left;\n" \
"            uint window = (py / size) * across + (px / size);\n" \
"            uint token = (py % size) * size + (px % size);\n" \
"            uint at = (window * size * size + token) * n + c;\n" \
"            value = ahalf ? (float)Ah[at] : A[at];\n" \
"        }\n" \
"        goto store;\n" \
"    }\n" \
"    case POOL2: {\n" \
"        uint width = sb; c = index % n; rest = index / n;\n" \
"        uint x = rest % (width / 2u), y = rest / (width / 2u);\n" \
"        uint base = ((2u * y) * width + 2u * x) * n + c;\n" \
"        float t = (ahalf ? (float)Ah[base] : A[base])\n" \
"                + (ahalf ? (float)Ah[base + width * n] : A[base + width * n]);\n" \
"        t += ahalf ? (float)Ah[base + n] : A[base + n];\n" \
"        t += ahalf ? (float)Ah[base + (width + 1u) * n] : A[base + (width + 1u) * n];\n" \
"        value = t * 0.25f; goto store;\n" \
"    }\n" \
"    case UPSAMPLE2: {\n" \
"        uint target = sa, source = sb; c = index % n; rest = index / n;\n" \
"        uint x = rest % target, y = rest / target;\n" \
"        uint at = ((y / 2u) * source + (x / 2u)) * n + c;\n" \
"        value = ahalf ? (float)Ah[at] : A[at]; goto store;\n" \
"    }\n" \
"    case UPSAMPLE_MERGE: {\n" \
"        uint target = sa, source = sb; c = index % n; rest = index / n;\n" \
"        uint x = rest % target, y = rest / target;\n" \
"        uint at = ((y / 2u) * source + (x / 2u)) * n + c;\n" \
"        float scaled = (ahalf ? (float)Ah[at] : A[at]) * D[c];\n" \
"        float merged = (bhalf ? (float)Bh[index] : B[index]) * D[n + c] + scaled;\n" \
"        C[index] = merged; Sh[index] = (half)e4m3(merged); return;\n" \
"    }\n" \
"    case POOL2_SKIP: {\n" \
"        uint width = sb; c = index % n; rest = index / n;\n" \
"        uint x = rest % (width / 2u), y = rest / (width / 2u);\n" \
"        uint base = ((2u * y) * width + 2u * x) * n + c;\n" \
"        uint taps[4]; taps[0] = base; taps[1] = base + width * n; taps[2] = base + n;\n" \
"        taps[3] = base + (width + 1u) * n;\n" \
"        float v0 = ahalf ? (float)Ah[taps[0]] : A[taps[0]];\n" \
"        float v1 = ahalf ? (float)Ah[taps[1]] : A[taps[1]];\n" \
"        float v2 = ahalf ? (float)Ah[taps[2]] : A[taps[2]];\n" \
"        float v3 = ahalf ? (float)Ah[taps[3]] : A[taps[3]];\n" \
"        float total = (v0 + v1) + v2 + v3;\n" \
"        Sh[taps[0]] = (half)e4m3(v0); Sh[taps[1]] = (half)e4m3(v1);\n" \
"        Sh[taps[2]] = (half)e4m3(v2); Sh[taps[3]] = (half)e4m3(v3);\n" \
"        /* through `store`: the pooled output is half (flag 0x1000) in the frame */\n" \
"        value = total * 0.25f; goto store;\n" \
"    }\n" \
"    case UPSAMPLE_ADD: {\n" \
"        uint target = sa, source = sb; c = index % n; rest = index / n;\n" \
"        uint x = rest % target, y = rest / target;\n" \
"        uint at = ((y / 2u) * source + (x / 2u)) * n + c;\n" \
"        float scaled = (bhalf ? (float)Bh[index] : B[index]) * D[c];\n" \
"        float merged = (ahalf ? (float)Ah[at] : A[at]) + scaled;\n" \
"        if (S != (__global float *)0) Sh[index] = (half)e4m3(merged);\n" \
"        value = merged; goto store;   /* honours the half output, as resident.comp */\n" \
"    }\n" \
"    case PAD_END: {\n" \
"        uint height = sa, width = sb, padded = sc; c = index % n; rest = index / n;\n" \
"        uint x = rest % padded, y = rest / padded;\n" \
"        uint at = (y * width + x) * n + c;\n" \
"        value = (y < height && x < width) ? (ahalf ? (float)Ah[at] : A[at]) : 0.0f;\n" \
"        goto store;\n" \
"    }\n" \
"    case SCALE_CHANNEL: value = (ahalf ? (float)Ah[index] : A[index]) * D[index % n]; goto store;\n" \
"    case ADD: value = (ahalf ? (float)Ah[index] : A[index])\n" \
"                    + (bhalf ? (float)Bh[index] : B[index]); goto store;\n" \
"    case SPLIT_HEADS: {\n" \
"        uint heads = batch, tokens = sa, part = sb;\n" \
"        uint d = index % 32u; rest = index / 32u;\n" \
"        uint token = rest % tokens; rest /= tokens;\n" \
"        uint head = rest % heads, window = rest / heads;\n" \
"        uint at = (window * tokens + token) * 3u * n + part * n + head * 32u + d;\n" \
"        value = ahalf ? (float)Ah[at] : A[at]; goto store;\n" \
"    }\n" \
"    case MERGE_HEADS: {\n" \
"        uint heads = batch, tokens = sa;\n" \
"        c = index % n; rest = index / n;\n" \
"        uint token = rest % tokens, window = rest / tokens;\n" \
"        uint at = (window * heads + c / 32u) * tokens + token;\n" \
"        value = ahalf ? (float)Ah[at * 32u + c % 32u] : A[at * 32u + c % 32u];\n" \
"        if ((flags & 0x20000u) != 0u) {\n" \
"            float scaled = HROUND(value) * D[at];\n" \
"            value = HROUND(scaled);\n" \
"        }\n" \
"        goto store;\n" \
"    }\n" \
"    case ADD_BIAS: {\n" \
"        uint plane = n * n, inside = index % plane;\n" \
"        value = (ahalf ? (float)Ah[index] : A[index])\n" \
"              + D[((index / plane) % batch) * plane + inside];\n" \
"        goto store;\n" \
"    }\n" \
"    default:\n" \
"        value = (ahalf ? (float)Ah[index] : A[index]) * p0;\n" \
"        if (kind == E4M3) value = e4m3(value);\n" \
"        else if (kind == GATE) value = gate_activation(value);\n" \
"        else if (kind == HALF) value = HROUND(value);\n" \
"        goto store;\n" \
"    }\n" \
"store:\n" \
"    value = publish_epi(epilogue, value);\n" \
"    if ((flags & 0x1000u) != 0u) Ch[index] = (half)value; else C[index] = value;\n" \
"}\n"

/* ---- attention.comp - the row-wise operators -------------------------------------- */
/* One invocation a row, the whole row in registers. The GLSL stages 32 rows through
 * shared memory only to make the global traffic contiguous; the arithmetic is per row,
 * so a row per invocation computes the same values (notes: cosine_tree's tree order and
 * the softmax's coupled pair transform are what must be reproduced, not the staging). */
#define COSINE_TREE \
"inline float hmul(float l, float r) { float t = l * r; return HROUND(t); }\n" \
"inline float hadd(float l, float r) { float t = l + r; return HROUND(t); }\n" \
"inline float hfma(float l, float r, float acc) { float t = l * r + acc; return HROUND(t); }\n" \
"#define COSINE_NORM_FLOOR 0.00006198883056640625f\n" \
"inline float cosine_reciprocal(float h[32]) {\n" \
"    float partial[4][2], two[4][2], one[4][2];\n" \
"    for (uint lane = 0u; lane < 4u; lane++)\n" \
"        for (uint parity = 0u; parity < 2u; parity++) {\n" \
"            uint ch = lane * 2u + parity;\n" \
"            float first = hfma(h[ch + 8u], h[ch + 8u], hmul(h[ch], h[ch]));\n" \
"            float second = hfma(h[ch + 24u], h[ch + 24u], hmul(h[ch + 16u], h[ch + 16u]));\n" \
"            partial[lane][parity] = hadd(first, second);\n" \
"        }\n" \
"    for (uint lane = 0u; lane < 4u; lane++)\n" \
"        for (uint p = 0u; p < 2u; p++) two[lane][p] = hadd(partial[lane][p], partial[lane ^ 2u][p]);\n" \
"    for (uint lane = 0u; lane < 4u; lane++)\n" \
"        for (uint p = 0u; p < 2u; p++) one[lane][p] = hadd(two[lane][p], two[lane ^ 1u][p]);\n" \
"    float norm = fmax(hadd(one[0][0], one[0][1]), HROUND(COSINE_NORM_FLOOR));\n" \
"    return HROUND(rsqrt(norm));\n" \
"}\n" \
"inline float vit_reciprocal(float h[32]) {\n" \
"    float r[16];\n" \
"    for (uint c = 0u; c < 16u; c++) r[c] = hfma(h[c], h[c], hmul(h[c + 16u], h[c + 16u]));\n" \
"    for (uint c = 0u; c < 8u; c++) r[c] = hadd(r[c], r[c + 8u]);\n" \
"    for (uint c = 0u; c < 4u; c++) r[c] = hadd(r[c], r[c + 4u]);\n" \
"    float norm = fmax(hadd(hadd(r[0], r[2]), hadd(r[1], r[3])), HROUND(COSINE_NORM_FLOOR));\n" \
"    return HROUND(1.0f / sqrt(norm));\n" \
"}\n" \
"#define VIT_ROOT 5.65625f\n"

#define ROW_SRC CL_COMMON COSINE_TREE \
"typedef union { uint u; half2 h; } u32h2;\n" \
"#define COSINE_PUBLISH 0u\n#define SOFTMAX 1u\n#define VIT_SOFTMAX 6u\n" \
"inline float vit_weight(float logit) {\n" \
"    float affine = HROUND(logit) * 0.08953857421875f;\n" \
"    affine += 1.708984375f;\n" \
"    affine = clamp(HROUND(affine), 1.439453125f, 1.9775390625f);\n" \
"    u32h2 c; c.h = (half2)(affine, 0.0f);\n" \
"    uint bits = c.u & 0xFFFFu;\n" \
"    c.u = ((bits << 4) + 0x4000u) & 0xFFFFu;\n" \
"    return (float)c.h.s0;\n" \
"}\n" \
"inline void pair_fast(float l0, float l1, float *w0, float *w1) {\n" \
"    float affine0 = clamp(HROUND(l0) * 0.044921875f + 1.30078125f, 1.03125f, 1.5693359375f);\n" \
"    float affine1 = clamp(HROUND(l1) * 0.044921875f + 1.30078125f, 1.03125f, 1.5693359375f);\n" \
"    u32h2 c; c.h = (half2)(affine0, affine1);\n" \
"    uint transformed = (c.u << 5) + 0x7FF88000u;\n" \
"    c.u = transformed;\n" \
"    *w0 = (float)c.h.s0; *w1 = (float)c.h.s1;\n" \
"}\n" \
"__kernel void row(uint flags, __global const float *A, __global const float *B,\n" \
"                  __global float *C, __global float *D,\n" \
"                  uint m, uint n, uint k, uint batch, uint sa, uint sb, uint sc, float p0)\n" \
"{\n" \
"    uint row = get_global_id(0);\n" \
"    if (row >= m) return;\n" \
"    uint kind = flags & 0xFFu;\n" \
"    __global half *Ch = (__global half *)C;\n" \
"    if (kind == COSINE_PUBLISH) {\n" \
"        uint qkv = flags & 0x20000u, part = (flags >> 18) & 1u;\n" \
"        float h[32];\n" \
"        for (uint i = 0u; i < 32u; i++) {\n" \
"            float v;\n" \
"            if (qkv) {\n" \
"                uint head = (row / n) % batch, token = row % n, window = row / (n * batch);\n" \
"                uint at = ((window * n + token) * 3u * batch + part * batch + head) * 32u;\n" \
"                v = A[at + i];\n" \
"            } else v = (flags & 0x8000u) ? (float)((__global const half *)A)[row * 32u + i]\n" \
"                                        : A[row * 32u + i];\n" \
"            h[i] = HROUND(v);\n" \
"        }\n" \
"        float reciprocal = (flags & 0x80000u) ? vit_reciprocal(h) : cosine_reciprocal(h);\n" \
"        float scale = 1.0f;\n" \
"        if (k) { uint head = (row / n) % batch; scale = HROUND(D[head]); }\n" \
"        for (uint i = 0u; i < 32u; i++) {\n" \
"            float value = hmul(h[i], reciprocal);\n" \
"            if (k) value = hmul((flags & 0x80000u) ? hmul(value, VIT_ROOT) : value, scale);\n" \
"            value = e4m3(value);\n" \
"            if (flags & 0x1000u) Ch[row * 32u + i] = (half)value; else C[row * 32u + i] = value;\n" \
"        }\n" \
"        return;\n" \
"    }\n" \
"    if (kind == VIT_SOFTMAX) {\n" \
"        uint stride = sa ? sa : n, tokens = n, base = row * stride;\n" \
"        uint blocks = (tokens + 63u) / 64u;\n" \
"        float zero = vit_weight(0.0f), total = 0.0f;\n" \
"        for (uint b = 0u; b < blocks; b++) {\n" \
"            float column[8];\n" \
"            for (uint c = 0u; c < 8u; c++) {\n" \
"                float pr[4];\n" \
"                for (uint j = 0u; j < 4u; j++) {\n" \
"                    uint k0 = b * 64u + c + 16u * j, k1 = k0 + 8u;\n" \
"                    float w0 = k0 < tokens ? vit_weight(A[base + k0]) : zero;\n" \
"                    float w1 = k1 < tokens ? vit_weight(A[base + k1]) : zero;\n" \
"                    pr[j] = hadd(w0, w1);\n" \
"                }\n" \
"                column[c] = hadd(hadd(hadd(pr[0], pr[1]), pr[2]), pr[3]);\n" \
"            }\n" \
"            float even = hadd(hadd(hadd(column[0], column[2]), column[4]), column[6]);\n" \
"            float odd = hadd(hadd(hadd(column[1], column[3]), column[5]), column[7]);\n" \
"            total = hadd(total, hadd(even, odd));\n" \
"        }\n" \
"        uint padding = blocks * 64u - tokens;\n" \
"        if (padding > 0u) total = hadd(total, -hmul(zero, (float)padding));\n" \
"        D[row] = HROUND(1.0f / total);\n" \
"        for (uint j = 0u; j < stride; j++) {\n" \
"            float value = j < tokens ? e4m3(vit_weight(A[base + j])) : 0.0f;\n" \
"            if (flags & 0x1000u) Ch[base + j] = (half)value; else C[base + j] = value;\n" \
"        }\n" \
"        return;\n" \
"    }\n" \
"    /* SOFTMAX: the rows are short enough here (<= 64) to recompute rather than stage. */\n" \
"    uint stride = sa ? sa : n, base = row * stride;\n" \
"    uint bias = ((row / n) % max(batch, 1u)) * n * n + (row % n) * n;\n" \
"    float total = 0.0f, w0, w1;\n" \
"    for (uint i = 0u; i < n; i += 2u) {\n" \
"        float l0 = A[base + i], l1 = A[base + i + 1u];\n" \
"        if (flags & 0x2000u) { l0 += B[bias + i]; l1 += B[bias + i + 1u]; }\n" \
"        if (p0 != 0.0f) { l0 = clamp(l0, -p0, p0); l1 = clamp(l1, -p0, p0); }\n" \
"        pair_fast(l0, l1, &w0, &w1);\n" \
"        total += w0; total += w1;\n" \
"    }\n" \
"    float reciprocal = HROUND(1.0f / HROUND(total));\n" \
"    for (uint i = 0u; i < stride; i++) {\n" \
"        float value = 0.0f;\n" \
"        if (i < n) {\n" \
"            float l0 = A[base + i];\n" \
"            if (flags & 0x2000u) l0 += B[bias + i];\n" \
"            if (p0 != 0.0f) l0 = clamp(l0, -p0, p0);\n" \
"            float ww0, ww1;\n" \
"            pair_fast(l0, 0.0f, &ww0, &ww1);\n" \
"            value = e4m3(hmul(ww0, reciprocal));\n" \
"        }\n" \
"        if (flags & 0x1000u) Ch[base + i] = (half)value; else C[base + i] = value;\n" \
"    }\n" \
"}\n"

/* ---- the QKV projection finished in its own epilogue (P3) -------------------------- */
/* The projection GEMM with Q and K cosine-normalised and V published before the block
 * leaves the workgroup, so the float32 projection never goes to memory and the two row
 * passes and the split that read it back are gone (the vendor's `qkv_epilogue.glsl`, worth
 * 21.6 % of a 720p frame there). Bit-identical to `xmx_rec_row`'s COSINE_PUBLISH for Q and
 * K and `xmx_rec_unary`'s SPLIT_HEADS for V: the raw accumulator is the value they would
 * have read back, the reciprocal norm is `COSINE_TREE`'s one definition, and the per-element
 * steps are theirs in their order.
 *
 * A workgroup is two subgroups (16 threads) covering one 32-column block - exactly one head
 * of one of Q, K or V, since the column block is 32 wide and 32-aligned. Each subgroup
 * computes 16 columns with the shipping FMA core (RM=2, RN=2, unchanged), stages the raw
 * accumulators to shared memory, and after a workgroup barrier one invocation per row
 * normalises its 32 channels in place. The GEMM arithmetic is therefore the shipped one. */
#define GEMM_QKV_SRC COSINE_TREE \
"#ifndef RM\n#define RM 2\n#endif\n" \
"#ifndef RN\n#define RN 2\n#endif\n" \
"#define QKV_COLS 32\n" \
"#define QKV_THREADS 16\n" \
"/* The epilogue reads the 32 staged channels straight from shared memory, so the norm's\n" \
" * operand never becomes a private array: this kernel already holds the GEMM's\n" \
" * accumulators, and a float[32] on top of them cost ~7 ms a frame. Same tree, same\n" \
" * rounding (HROUND is pure, so rounding each read matches rounding once), `__local`. */\n" \
"#define QKV_HR(i) HROUND(h[i])\n" \
"inline float cosine_reciprocal_local(__local const float *h) {\n" \
"    float partial[4][2], two[4][2], one[4][2];\n" \
"    for (uint lane = 0u; lane < 4u; lane++)\n" \
"        for (uint parity = 0u; parity < 2u; parity++) {\n" \
"            uint ch = lane * 2u + parity;\n" \
"            float first = hfma(QKV_HR(ch + 8u), QKV_HR(ch + 8u), hmul(QKV_HR(ch), QKV_HR(ch)));\n" \
"            float second = hfma(QKV_HR(ch + 24u), QKV_HR(ch + 24u), hmul(QKV_HR(ch + 16u), QKV_HR(ch + 16u)));\n" \
"            partial[lane][parity] = hadd(first, second);\n" \
"        }\n" \
"    for (uint lane = 0u; lane < 4u; lane++)\n" \
"        for (uint p = 0u; p < 2u; p++) two[lane][p] = hadd(partial[lane][p], partial[lane ^ 2u][p]);\n" \
"    for (uint lane = 0u; lane < 4u; lane++)\n" \
"        for (uint p = 0u; p < 2u; p++) one[lane][p] = hadd(two[lane][p], two[lane ^ 1u][p]);\n" \
"    float norm = fmax(hadd(one[0][0], one[0][1]), HROUND(COSINE_NORM_FLOOR));\n" \
"    return HROUND(rsqrt(norm));\n" \
"}\n" \
"inline float vit_reciprocal_local(__local const float *h) {\n" \
"    float r[16];\n" \
"    for (uint c = 0u; c < 16u; c++) r[c] = hfma(QKV_HR(c), QKV_HR(c), hmul(QKV_HR(c + 16u), QKV_HR(c + 16u)));\n" \
"    for (uint c = 0u; c < 8u; c++) r[c] = hadd(r[c], r[c + 8u]);\n" \
"    for (uint c = 0u; c < 4u; c++) r[c] = hadd(r[c], r[c + 4u]);\n" \
"    float norm = fmax(hadd(hadd(r[0], r[2]), hadd(r[1], r[3])), HROUND(COSINE_NORM_FLOOR));\n" \
"    return HROUND(1.0f / sqrt(norm));\n" \
"}\n" \
"__attribute__((intel_reqd_sub_group_size(8)))\n" \
"__kernel void gemm_qkv(__global const half *A, __global const half *B,\n" \
"                       __global half *Q, __global half *Kt, __global half *V,\n" \
"                       __global const float *Scale,\n" \
"                       unsigned M, unsigned N, unsigned K,\n" \
"                       unsigned lda, unsigned ldb, unsigned oa, unsigned ob,\n" \
"                       unsigned tokens, unsigned heads, unsigned vit)\n" \
"{\n" \
"    __local float stg[8 * RM][QKV_COLS + 1];   /* +1 float so a row read is conflict-free */\n" \
"    unsigned lid = get_local_id(0), sg = lid >> 3u, lane = lid & 7u;\n" \
"    unsigned row0 = get_group_id(1) * (8 * RM);\n" \
"    unsigned col0 = get_group_id(0) * QKV_COLS;\n" \
"    A += oa; B += ob;\n" \
"    unsigned col = col0 + sg * (8 * RN);\n" \
"    float8 acc[RM][RN];\n" \
"#pragma unroll\n" \
"    for (unsigned r = 0; r < RM; r++)\n" \
"#pragma unroll\n" \
"        for (unsigned c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);\n" \
"    for (unsigned k = 0; k < K; k += 8) {\n" \
"        float bv[RN][8];\n" \
"#pragma unroll\n" \
"        for (unsigned c = 0; c < RN; c++) {\n" \
"            unsigned bcol = col + c * 8 + lane;\n" \
"#pragma unroll\n" \
"            for (unsigned j = 0; j < 8; j++)\n" \
"                bv[c][j] = (float)B[(size_t)(k + j) * ldb + bcol];\n" \
"        }\n" \
"#pragma unroll\n" \
"        for (unsigned r = 0; r < RM; r++)\n" \
"#pragma unroll\n" \
"            for (unsigned i = 0; i < 8; i++) {\n" \
"                float4 lo = vload_half4(0, A + (size_t)(row0 + r * 8 + i) * lda + k);\n" \
"                float4 hi = vload_half4(0, A + (size_t)(row0 + r * 8 + i) * lda + k + 4);\n" \
"                float av[8] = { lo.s0, lo.s1, lo.s2, lo.s3, hi.s0, hi.s1, hi.s2, hi.s3 };\n" \
"#pragma unroll\n" \
"                for (unsigned c = 0; c < RN; c++) {\n" \
"                    float s = acc[r][c][i];\n" \
"#pragma unroll\n" \
"                    for (unsigned j = 0; j < 8; j++) s = mad(av[j], bv[c][j], s);\n" \
"                    acc[r][c][i] = s;\n" \
"                }\n" \
"            }\n" \
"    }\n" \
"#pragma unroll\n" \
"    for (unsigned r = 0; r < RM; r++)\n" \
"#pragma unroll\n" \
"        for (unsigned i = 0; i < 8; i++)\n" \
"#pragma unroll\n" \
"            for (unsigned c = 0; c < RN; c++)\n" \
"                stg[r * 8 + i][sg * (8 * RN) + c * 8 + lane] = acc[r][c][i];\n" \
"    barrier(CLK_LOCAL_MEM_FENCE);\n" \
"    unsigned chunk = col0 >> 5u;\n" \
"    unsigned part = chunk / heads, head = chunk % heads;\n" \
"    for (unsigned lr = lid; lr < 8 * RM; lr += QKV_THREADS) {\n" \
"        unsigned m = row0 + lr;\n" \
"        unsigned at = (((m / tokens) * heads + head) * tokens + (m % tokens)) * 32u;\n" \
"        if (part == 2u) {\n" \
"            for (unsigned d = 0; d < 32u; d++) V[at + d] = (half)e4m3(stg[lr][d]);\n" \
"            continue;\n" \
"        }\n" \
"        float reciprocal = vit ? vit_reciprocal_local(stg[lr])\n" \
"                              : cosine_reciprocal_local(stg[lr]);\n" \
"        float scale = part == 0u ? HROUND(Scale[head]) : 1.0f;\n" \
"        __global half *dst = part == 0u ? Q : Kt;\n" \
"        for (unsigned d = 0; d < 32u; d++) {\n" \
"            float value = hmul(HROUND(stg[lr][d]), reciprocal);\n" \
"            if (part == 0u) value = hmul(vit ? hmul(value, VIT_ROOT) : value, scale);\n" \
"            value = e4m3(value);\n" \
"            dst[at + d] = (half)value;\n" \
"        }\n" \
"    }\n" \
"}\n"

/* ---- window attention finished on-chip (P3) ---------------------------------------- */
/* One workgroup per (window, head), eight SIMD8 subgroups; subgroup s owns query rows
 * 8s..8s+7 and every key. Lane l owns the key pairs (2l, 2l+1) and (16+2l, 16+2l+1) of
 * each 32-key half, so the coupled `pair_fast` transform runs inside the lane; the
 * shipping FMA core's 8-wide K tiling is reproduced inside (a uniform Q row against four
 * staged K keys), so K is read once and reused across eight rows instead of reloaded per
 * row (Run 33's row-per-lane kernel lost for exactly that reason). The scores and
 * probabilities never leave shared memory; the row SOFTMAX's coupled denominator and the
 * uncoupled per-key probabilities are reproduced bit for bit. */
#define ATTN_SRC CL_COMMON COSINE_TREE \
"typedef union { uint u; half2 h; } attn_h2;\n" \
"inline float attn_affine(float l) {\n" \
"    return clamp(HROUND(l) * 0.044921875f + 1.30078125f, 1.03125f, 1.5693359375f);\n" \
"}\n" \
"inline float attn_weight(uint packed_a) {\n" \
"    attn_h2 c; c.u = 0u; c.u = ((packed_a << 5) + 0x8000u) & 0xFFFFu;\n" \
"    return (float)c.h.s0;\n" \
"}\n" \
"__attribute__((intel_reqd_sub_group_size(8)))\n" \
"__kernel void attn(__global const half *Q, __global const half *Kb, __global const half *Vb,\n" \
"                   __global const float *Bias, __global float *Ctx, __global half *Out,\n" \
"                   uint batches, uint heads, uint merged, uint bias_on)\n" \
"{\n" \
"    uint b = get_group_id(0);\n" \
"    if (b >= batches) return;\n" \
"    uint head = b % heads, window = b / heads;\n" \
"    size_t base = (size_t)b * 2048u;\n" \
"    uint sub = get_local_id(0) >> 3u, lane = get_sub_group_local_id();\n" \
"    __local uint sm[64][33];\n" \
"    for (uint hh = 0u; hh < 2u; hh++) {\n" \
"        float acc[8][4];\n" \
"        for (uint i = 0u; i < 8u; i++)\n" \
"            for (uint q = 0u; q < 4u; q++) acc[i][q] = 0.0f;\n" \
"        uint key[4];\n" \
"        key[0] = 32u * hh + 2u * lane; key[1] = key[0] + 1u;\n" \
"        key[2] = 32u * hh + 16u + 2u * lane; key[3] = key[2] + 1u;\n" \
"        for (uint k = 0u; k < 32u; k += 8u) {\n" \
"            float kv[4][8];\n" \
"            for (uint q = 0u; q < 4u; q++) {\n" \
"                float8 t = vload_half8(0, Kb + base + (size_t)key[q] * 32u + k);\n" \
"                kv[q][0] = t.s0; kv[q][1] = t.s1; kv[q][2] = t.s2; kv[q][3] = t.s3;\n" \
"                kv[q][4] = t.s4; kv[q][5] = t.s5; kv[q][6] = t.s6; kv[q][7] = t.s7;\n" \
"            }\n" \
"            for (uint i = 0u; i < 8u; i++) {\n" \
"                float8 qv = vload_half8(0, Q + base + (size_t)(8u * sub + i) * 32u + k);\n" \
"                for (uint q = 0u; q < 4u; q++) {\n" \
"                    float s = acc[i][q];\n" \
"                    s = mad(qv.s0, kv[q][0], s); s = mad(qv.s1, kv[q][1], s);\n" \
"                    s = mad(qv.s2, kv[q][2], s); s = mad(qv.s3, kv[q][3], s);\n" \
"                    s = mad(qv.s4, kv[q][4], s); s = mad(qv.s5, kv[q][5], s);\n" \
"                    s = mad(qv.s6, kv[q][6], s); s = mad(qv.s7, kv[q][7], s);\n" \
"                    acc[i][q] = s;\n" \
"                }\n" \
"            }\n" \
"        }\n" \
"        for (uint i = 0u; i < 8u; i++) {\n" \
"            uint token = 8u * sub + i;\n" \
"            size_t bat = (size_t)head * 4096u + (size_t)token * 64u;\n" \
"            for (uint c2 = 0u; c2 < 2u; c2++) {\n" \
"                uint pa = 32u * hh + 16u * c2 + 2u * lane;\n" \
"                float l0 = acc[i][2u * c2], l1 = acc[i][2u * c2 + 1u];\n" \
"                if (bias_on) { l0 += Bias[bat + pa]; l1 += Bias[bat + pa + 1u]; }\n" \
"                attn_h2 c; c.h = (half2)(attn_affine(l0), attn_affine(l1));\n" \
"                sm[token][16u * hh + 8u * c2 + lane] = c.u;\n" \
"            }\n" \
"        }\n" \
"    }\n" \
"    sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n" \
"    {\n" \
"        uint token = 8u * sub + lane;\n" \
"        float total = 0.0f;\n" \
"        for (uint p = 0u; p < 32u; p++) {\n" \
"            attn_h2 c; c.u = (sm[token][p] << 5) + 0x7FF88000u;\n" \
"            total += (float)c.h.s0; total += (float)c.h.s1;\n" \
"        }\n" \
"        float reciprocal = HROUND(1.0f / HROUND(total));\n" \
"        for (uint p = 0u; p < 32u; p++) {\n" \
"            uint packed = sm[token][p];\n" \
"            float e = e4m3(hmul(attn_weight(packed & 0xFFFFu), reciprocal));\n" \
"            float o = e4m3(hmul(attn_weight(packed >> 16), reciprocal));\n" \
"            attn_h2 c; c.h = (half2)(e, o);\n" \
"            sm[token][p] = c.u;\n" \
"        }\n" \
"    }\n" \
"    sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n" \
"    {\n" \
"        float acc[8][4];\n" \
"        for (uint i = 0u; i < 8u; i++)\n" \
"            for (uint c = 0u; c < 4u; c++) acc[i][c] = 0.0f;\n" \
"        for (uint jb = 0u; jb < 64u; jb += 4u) {\n" \
"            float vv[4][4];\n" \
"            for (uint jj = 0u; jj < 4u; jj++) {\n" \
"                float4 t = vload_half4(0, Vb + base + (size_t)(jb + jj) * 32u + 4u * lane);\n" \
"                vv[jj][0] = t.s0; vv[jj][1] = t.s1; vv[jj][2] = t.s2; vv[jj][3] = t.s3;\n" \
"            }\n" \
"            for (uint i = 0u; i < 8u; i++) {\n" \
"                __local const half *pr = (__local const half *)&sm[8u * sub + i][jb >> 1];\n" \
"                float4 pv = vload_half4(0, pr);\n" \
"                for (uint c = 0u; c < 4u; c++) {\n" \
"                    float s = acc[i][c];\n" \
"                    s = mad(pv.s0, vv[0][c], s); s = mad(pv.s1, vv[1][c], s);\n" \
"                    s = mad(pv.s2, vv[2][c], s); s = mad(pv.s3, vv[3][c], s);\n" \
"                    acc[i][c] = s;\n" \
"                }\n" \
"            }\n" \
"        }\n" \
"        for (uint i = 0u; i < 8u; i++) {\n" \
"            uint token = 8u * sub + i;\n" \
"            for (uint c = 0u; c < 4u; c++) {\n" \
"                uint d = 4u * lane + c;\n" \
"                if (merged) Out[(((size_t)window * 64u + token) * heads + head) * 32u + d]\n" \
"                    = (half)e4m3(acc[i][c]);\n" \
"                else Ctx[((size_t)b * 64u + token) * 32u + d] = acc[i][c];\n" \
"            }\n" \
"        }\n" \
"    }\n" \
"}\n"

/* ---- runtime state ---------------------------------------------------------------- */
#define MAX_BUF 8192
#define MAX_GRAPHS 64
#define MAX_CMD 65536
#define K_GEMM 0
#define K_COPY 1
#define K_UNARY 2
#define K_UNARY2 3
#define K_ROW 4
#define K_COMPOSED 5
#define K_GEMM_QKV 6
#define K_ATTN 7

struct buffer {
	int used;
	void *host;                 /* non-NULL only for a mapped buffer: the memory `mem` wraps */
	cl_mem mem;
	unsigned long long bytes;
	int kind;                   /* GRAPH=0 (device-local, unmapped), HOST_READ/WRITE != 0 */
	int internal;               /* the composed passes' own scratch, not the model's */
	/* VNNI-packed copy of this buffer as a GEMM's B operand, and the generation it was
	 * packed at. `gen` bumps whenever the buffer is written (host upload or a command),
	 * so a weight packs once and an activation-B repacks when it changes. */
	cl_mem packed;              /* NULL until first use as a packed B */
	unsigned long long gen;     /* bumped whenever the buffer is written */
	unsigned long long packed_gen;   /* the gen the packed copy was made at */
	unsigned packed_K, packed_N, packed_stride, packed_batch;
};

struct cmd {
	int kind;
	int a, b, c, d, second;      /* d = third operand; second = a pass's second output */
	unsigned M, N, K, batch, sa, sb, sc, bt, lda, ldb, ldc, oa, ob, oc;
	unsigned n, channels, p0, k;  /* elementwise: count, channels, scale bits, pad */
	unsigned rows, width, heads, scaled, stride;
	unsigned height, across, pad, flags;
	unsigned long long bytes, so, to;
	float cap;
	int stamp;                   /* profile kind, family * 32 + subkind (libxmx's numbering) */
	struct cmd *parts;           /* a composed op's lowered commands (K_COMPOSED) */
	int nparts;
};
static struct cmd *new_cmd(int kind);
static void free_parts(struct cmd *list, int n);
/* The descriptor path's three operands: host shadows the caller writes through
 * (`xmx_reserve*` returns them) mirrored to device buffers, as libxmx's A/B/C are. */
static void *g_ra, *g_rb, *g_rc;
static int g_ria = -1, g_rib = -1, g_ric = -1;
static unsigned long long g_rasz, g_rbsz, g_rcsz;

static struct {
	int open;
	cl_context ctx;
	cl_command_queue q;
	cl_device_id dev;
	cl_program prog;
	cl_kernel gemm;
	cl_kernel gemm_small;       /* RM=1: one 8-row tile, for M that is not a multiple of 16 */
	cl_kernel gemm_blk;         /* block-read A + VNNI-packed B (P1) */
	cl_kernel pack;             /* B -> VNNI fragments for gemm_blk */
	cl_kernel gemm_qkv;         /* the QKV projection with its epilogue fused (P3) */
	cl_kernel attn;             /* the window attention finished on-chip (P3) */
	cl_kernel resident;
	cl_kernel row;
	int rm, rn;
	struct buffer buf[MAX_BUF];
	struct cmd cmd[MAX_CMD];
	int ncmd;
	struct { int live; struct cmd *list; int n; } graph[MAX_GRAPHS];
	char error[512];
	char device[256];
	int coopmat_n;
	unsigned long long total;
	char *spv_cache;            /* xmx_reserve hands back device addresses; see below */
	int time;                   /* XMX_TIME: collect per-dispatch profiling events */
	int qprof;                  /* the command queue carries CL_QUEUE_PROFILING_ENABLE */
	int prof;                   /* xmx_profile(1): fill the per-kind table from the events */
	unsigned long long dev_ns;  /* summed device time of the last execute() */
	int dev_cmds;               /* dispatches profiled in the last execute() */
} g;

static struct { cl_event ev; int tag; int kind; int group; } *g_trace;
static int g_ntrace;
static int g_top_stamp;    /* the profile kind of the top-level command being executed */
static int g_group;        /* which top-level command a pushed event belongs to */

/* The profile table, the same shape and kind numbering as libxmx: a kind is
 * `family * 32 + subkind`, families PK_GEMM/TILED/STAGED/UNARY/ROW/HISTORY/COPY. */
enum { PK_GEMM = 0, PK_TILED, PK_STAGED, PK_UNARY, PK_ROW, PK_HISTORY, PK_COPY, PK_START = 7 };
#define PROF_KINDS (PK_START * 32)
#define PROF_EACH_MAX 8192
#define PK(f, s) ((f) * 32 + (s))
static double prof_ms[PROF_KINDS];
static unsigned prof_hits[PROF_KINDS];
static int prof_each_n;
static double prof_each[PROF_EACH_MAX];
static unsigned prof_each_kind[PROF_EACH_MAX];

static int profiling(void) { return g.time || g.prof; }

static void trace_reset(void)
{
	g_group = 0;
	g_top_stamp = 0;
	if (!profiling())
		return;
	for (int i = 0; i < g_ntrace; i++)
		clReleaseEvent(g_trace[i].ev);
	g_ntrace = 0;
}

static void trace_push(cl_event ev, int tag)
{
	if (profiling() && ev) {
		g_trace[g_ntrace].ev = ev;
		g_trace[g_ntrace].tag = tag;
		g_trace[g_ntrace].kind = g_top_stamp;
		g_trace[g_ntrace].group = g_group;
		g_ntrace++;
	}
}

/* One recorded pass per `xmx_rec_*` call, as libxmx reports, not one per kernel this
 * backend lowers a composed op into: the events of one top-level command are summed into
 * a single entry carrying that command's kind, so the per-pass table and the recorded
 * labels line up. */
static void collect_profile(void)
{
	int i = 0;
	while (i < g_ntrace) {
		int group = g_trace[i].group, kind = g_trace[i].kind;
		double ms = 0.0;
		int any = 0;
		while (i < g_ntrace && g_trace[i].group == group) {
			cl_ulong s = 0, e = 0;
			if (clGetEventProfilingInfo(g_trace[i].ev, CL_PROFILING_COMMAND_START,
						    sizeof s, &s, NULL) == CL_SUCCESS
			    && clGetEventProfilingInfo(g_trace[i].ev, CL_PROFILING_COMMAND_END,
						       sizeof e, &e, NULL) == CL_SUCCESS && e > s) {
				ms += (double)(e - s) * 1e-6;
				any = 1;
			}
			i++;
		}
		if (!any || kind < 0 || kind >= PROF_KINDS)
			continue;
		if (prof_each_n < PROF_EACH_MAX) {
			prof_each_kind[prof_each_n] = (unsigned)kind;
			prof_each[prof_each_n++] = ms;
		}
		prof_ms[kind] += ms;
		prof_hits[kind]++;
	}
}

static void trace_sum(void)
{
	if (g.prof)
		collect_profile();
	if (!g.time) {
		trace_reset();
		return;
	}
	g.dev_ns = 0;
	g.dev_cmds = g_ntrace;
	int verbose = getenv("XMX_TIME_VERBOSE") != NULL;
	double fam[5] = {0};
	int famn[5] = {0};
	static const char *names[5] = {"gemm", "unary", "unary2", "row", "copy"};
	for (int i = 0; i < g_ntrace; i++) {
		cl_ulong s = 0, e = 0;
		clGetEventProfilingInfo(g_trace[i].ev, CL_PROFILING_COMMAND_START, sizeof s, &s, NULL);
		clGetEventProfilingInfo(g_trace[i].ev, CL_PROFILING_COMMAND_END, sizeof e, &e, NULL);
		if (e > s) {
			g.dev_ns += e - s;
			int t = g_trace[i].tag;
			fam[t] += (e - s) / 1e6;
			famn[t]++;
		}
		if (verbose)
			fprintf(stderr, "XMX_TRACE %d %d %.3f ms\n", i, g_trace[i].tag, (e - s) / 1e6);
		clReleaseEvent(g_trace[i].ev);
	}
	fprintf(stderr, "XMX_DEVTIME %.3f ms in %d dispatches:", g.dev_ns / 1e6, g.dev_cmds);
	for (int t = 0; t < 5; t++)
		fprintf(stderr, " %s=%.1f/%d", names[t], fam[t], famn[t]);
	fprintf(stderr, "\n");
	g_ntrace = 0;
}

static int fail(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(g.error, sizeof g.error, fmt, ap);
	va_end(ap);
	fprintf(stderr, "openclmx: %s\n", g.error);
	return -1;
}

int xmx_device_lost(void) { return 0; }
const char *xmx_error(void) { return g.error; }
const char *xmx_device(void) { return g.device; }
const char *xmx_path(void) { return "libOpenCL"; }
const char *xmx_memory(void) { return "card memory, not host-visible: the host reaches it by copies"; }
int xmx_staging_mode(void) { return 1; }
int xmx_discrete(void) { return 1; }
int xmx_preserve16(void) { return 1; }
int xmx_half_by_cast(void) { return 1; }
int xmx_staged_packed(void) { return 1; }
/* ponytail: one A loader here, so XMX_PORTABLE_PACKED is echoed but selects nothing;
 * test_staged_packed then checks the two modes byte-identical. Make it real if a packed
 * loader variant is added. */
int xmx_portable_packed(void) { const char *e = getenv("XMX_PORTABLE_PACKED"); return e && atoi(e); }
int xmx_adopted(void) { return 0; }
int xmx_coopmat(void) { return 1; }
int xmx_portable(void) { return 0; }
int xmx_coopmat_n(void) { return g.coopmat_n; }
int xmx_coopmat_int8_n(void) { return 0; }
int xmx_window_gather(void) { return 0; }
size_t xmx_embedded_shader(const char *) { return 0; }
void xmx_cancel(int on) { (void)on; }
int xmx_cancelled(void) { return 0; }
int xmx_specialize(unsigned mask) { (void)mask; return 0; }
unsigned xmx_specialized_count(void) { return 0; }
unsigned xmx_specialization(void) { return 0; }

static cl_kernel build_kernel(const char *src, const char *opts, const char *name)
{
	cl_int err;
	const char *list[1] = { src };
	cl_program prog = clCreateProgramWithSource(g.ctx, 1, list, NULL, &err);
	if (err != CL_SUCCESS)
		return NULL;
	/* XMX_CL_OPTS is appended to every build: a diagnostic for compiler faults */
	char all[512];
	snprintf(all, sizeof all, "%s %s", opts ? opts : "", getenv("XMX_CL_OPTS") ? getenv("XMX_CL_OPTS") : "");
	if (clBuildProgram(prog, 1, &g.dev, all, NULL, NULL) != CL_SUCCESS) {
		size_t n = 0;
		char log[8192];
		clGetProgramBuildInfo(prog, g.dev, 0x1183 /*CL_PROGRAM_BUILD_LOG*/, sizeof log, log, &n);
		fail("kernel build failed: %.900s", log);
		return NULL;
	}
	return clCreateKernel(prog, name, &err);
}

int xmx_open(void)
{
	cl_platform_id plats[8];
	cl_uint nplat = 0, ndev = 0;
	cl_int err;
	g.ncmd = 0;
	g.total = 0;
	if (clGetPlatformIDs(8, plats, &nplat) != CL_SUCCESS || nplat == 0)
		return fail("no OpenCL platform");
	for (cl_uint p = 0; p < nplat; p++) {
		if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &g.dev, &ndev) == CL_SUCCESS && ndev)
			break;
	}
	if (!g.dev)
		return fail("no OpenCL GPU device");
	g.ctx = clCreateContext(NULL, 1, &g.dev, NULL, NULL, &err);
	if (!g.ctx)
		return fail("clCreateContext: %d", err);
	g.time = getenv("XMX_TIME") != NULL;
	g.qprof = g.time;
	g.q = clCreateCommandQueue(g.ctx, g.dev, g.qprof ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
	if (!g.q)
		return fail("clCreateCommandQueue: %d", err);
	if (g.time)
		g_trace = malloc(sizeof(*g_trace) * MAX_CMD);
	g.rm = 2;
	g.rn = 2;
	g.coopmat_n = 8;
	{
		size_t n = strlen(CL_COMMON) + strlen(GEMM_SRC) + 1;
		char *src = malloc(n);
		if (!src)
			return -1;
		snprintf(src, n, "%s%s", CL_COMMON, GEMM_SRC);
		if (getenv("XMX_DUMP_SRC")) {
			FILE *f = fopen(getenv("XMX_DUMP_SRC"), "w");
			if (f) { fputs(src, f); fclose(f); }
		}
		/* Built optimised, the same as HEAD. A run-to-run difference here is a memory
		 * bug, not the compiler: every buffer is cleared at creation (see
		 * xmx_buf_create_kind). XMX_CL_GEMM_OPTS overrides the flags for diagnosis. */
		const char *gx = getenv("XMX_CL_GEMM_OPTS") ? getenv("XMX_CL_GEMM_OPTS") : "";
		char o2[256], o1[256];
		snprintf(o2, sizeof o2, "-D RN=%d %s", g.rn, gx);
		snprintf(o1, sizeof o1, "-D RM=1 -D RN=%d %s", g.rn, gx);
		g.gemm = build_kernel(src, o2, "gemm");
		g.gemm_small = build_kernel(src, o1, "gemm");
		free(src);
	}
	if (!g.gemm)
		return -1;
	/* The fused QKV-epilogue projection (P3). A build failure is not fatal: xmx_rec_gemm_qkv
	 * falls back to the decomposed passes. XMX_CL_GEMM_OPTS is shared with the GEMM. */
	{
		size_t n = strlen(CL_COMMON) + strlen(GEMM_QKV_SRC) + 1;
		char *src = malloc(n);
		if (src) {
			snprintf(src, n, "%s%s", CL_COMMON, GEMM_QKV_SRC);
			const char *gx = getenv("XMX_CL_GEMM_OPTS") ? getenv("XMX_CL_GEMM_OPTS") : "";
			char oq[256];
			snprintf(oq, sizeof oq, "-D RM=2 -D RN=%d %s", g.rn, gx);
			g.gemm_qkv = build_kernel(src, oq, "gemm_qkv");
			if (g.gemm_qkv)
				g.error[0] = 0;   /* not fatal: the decomposed path stays available */
			free(src);
		}
	}
	/* The block-read GEMM and its packer. P1's fast path; the packer is authoritative only
	 * when the buffers it names allow it (run_gemm's guards). XMX_CL_GEMM_OPTS overrides
	 * the flags for diagnosis. */
	if (getenv("XMX_CL_BLK")) {
		size_t n = strlen(CL_COMMON) + strlen(GEMM_BLK_SRC) + 1;
		char *blk = malloc(n), *pk = malloc(strlen(CL_COMMON) + strlen(PACK_SRC) + 1);
		if (!blk || !pk)
			return -1;
		snprintf(blk, n, "%s%s", CL_COMMON, GEMM_BLK_SRC);
		snprintf(pk, strlen(CL_COMMON) + strlen(PACK_SRC) + 1, "%s%s", CL_COMMON, PACK_SRC);
		const char *bx = getenv("XMX_CL_GEMM_OPTS") ? getenv("XMX_CL_GEMM_OPTS") : "";
		g.gemm_blk = build_kernel(blk, bx, "gemm");
		g.pack = build_kernel(pk, NULL, "pack");
		free(blk);
		free(pk);
		if (g.gemm_blk && g.pack)
			g.error[0] = 0;   /* build_kernel's failure is not fatal: the scalar path stays */
	}
	/* The window attention finished on-chip (P3). A build failure is not fatal:
	 * xmx_rec_window_attention keeps the decomposed passes. */
	g.attn = build_kernel(ATTN_SRC, NULL, "attn");
	if (g.attn)
		g.error[0] = 0;
	g.resident = build_kernel(RESIDENT_SRC, NULL, "resident");
	if (!g.resident)
		return -1;
	g.row = build_kernel(ROW_SRC, NULL, "row");
	if (!g.row)
		return -1;
	snprintf(g.device, sizeof g.device, "Intel(R) OpenCL (DG2, SIMD8 DPAS)");
	g.open = 1;
	return 0;
}

int xmx_buf_destroy(int id);
int xmx_graph_destroy(int id);
static void scratch_reset(void);
static void collect_scratch(void);

/* As libxmx's: every graph and buffer released, so a reopen starts clean and nothing the
 * device holds outlives the context's user. */
void xmx_close(void)
{
	if (!g.open)
		return;
	for (int i = 0; i < MAX_GRAPHS; i++)
		xmx_graph_destroy(i);
	for (int i = 0; i < MAX_BUF; i++)
		xmx_buf_destroy(i);
	free(g_ra); g_ra = NULL; g_ria = -1; g_rasz = 0;
	free(g_rb); g_rb = NULL; g_rib = -1; g_rbsz = 0;
	free(g_rc); g_rc = NULL; g_ric = -1; g_rcsz = 0;
	free_parts(g.cmd, g.ncmd);
	scratch_reset();
	g.ncmd = 0;
	g.open = 0;
}

int xmx_adopt(void *instance, void *physical_device, void *device, void *queue,
	      unsigned queue_family, int cooperative_matrix, void *get_instance_proc_addr,
	      void (*lock)(void *), void (*release)(void *), void *lock_context)
{
	(void)instance; (void)physical_device; (void)device; (void)queue; (void)queue_family;
	(void)cooperative_matrix; (void)get_instance_proc_addr; (void)lock; (void)release;
	(void)lock_context;
	return fail("xmx_adopt is not supported by the OpenCL backend");
}

/* ---- buffers ---------------------------------------------------------------------- */
static struct buffer *slot(int id)
{
	if (id < 0 || id >= MAX_BUF || !g.buf[id].used)
		return NULL;
	return &g.buf[id];
}

/* On a card whose memory the host cannot see (the only kind with a CL_MEM that is both
 * device-local and fast), there is no OpenCL buffer the host can write through and the
 * device then reads - `CL_MEM_USE_HOST_PTR` is copied at creation and ignores every later
 * host write, which is why the first basic check came back all zeros. So the mapped kinds
 * are a host shadow pushed and pulled at submit: HOST_WRITE is the host's and goes up,
 * HOST_READ is the device's and comes down. GRAPH buffers never leave the card; the host
 * reaches them only through upload/download. This is phase65's split, emulated. */
static int fill_device(unsigned id)
{
	struct buffer *b = slot(id);
	static const unsigned char zeros[64] = { 0 };
	unsigned long long words = b->bytes & ~3ull;
	if (words && clEnqueueFillBuffer(g.q, b->mem, zeros, 4, 0, (size_t)words, 0, NULL, NULL) != CL_SUCCESS)
		return fail("clear buffer: %d", -1);
	if (words < b->bytes
	    && clEnqueueWriteBuffer(g.q, b->mem, CL_TRUE, (size_t)words, (size_t)(b->bytes - words),
				    zeros, 0, NULL, NULL) != CL_SUCCESS)
		return fail("clear buffer tail: %d", -1);
	return 0;
}

int xmx_buf_create_kind(unsigned long long bytes, int kind)
{
	cl_int err;
	if (!g.open)
		return fail("device not open");
	for (int i = 0; i < MAX_BUF; i++) {
		if (g.buf[i].used)
			continue;
		size_t alloc = bytes ? bytes : 1;
		void *host = NULL;
		if (kind != 0) {
			host = malloc(alloc);
			if (!host)
				return fail("out of memory for a mapped buffer");
			memset(host, 0, alloc);
		}
		cl_mem mem = clCreateBuffer(g.ctx, CL_MEM_READ_WRITE, alloc, NULL, &err);
		if (!mem) {
			free(host);
			return fail("clCreateBuffer: %d", err);
		}
		g.buf[i].used = 1;
		g.buf[i].host = host;
		g.buf[i].mem = mem;
		g.buf[i].bytes = bytes;
		g.buf[i].kind = kind;
		g.buf[i].internal = 0;
		/* Clear the device buffer for every kind: a kernel that reads a pad a previous
		 * pass never wrote (a tail, a masked row) otherwise carries whatever the driver's
		 * fresh allocation held, which is not the same twice, and shows up as a pass that
		 * differs run to run (`test_determinism.py`). This is the same reason
		 * `nr_resident.py` zeroes its scratch. */
		if (fill_device(i)) {
			g.buf[i].used = 0;
			clReleaseMemObject(mem);
			free(host);
			return -1;
		}
		return i;
	}
	return fail("buffer table full");
}

int xmx_buf_create(unsigned long long bytes) { return xmx_buf_create_kind(bytes, 0); }

int xmx_buf_host_visible(int id) { struct buffer *b = slot(id); return (b && b->host) ? 1 : 0; }
void *xmx_buf_ptr(int id) { struct buffer *b = slot(id); return b ? b->host : NULL; }
unsigned long long xmx_buf_bytes(int id) { struct buffer *b = slot(id); return b ? b->bytes : 0; }
/* The model's live resident buffers, as libd3dmx counts them: summed from the table, so a
 * destroy is counted (a running total that only grew made every close look like a leak).
 * The composed passes' scratch is the backend's own, like a driver allocation, and is
 * left out - the fused kernels that replace them (P3) have none. */
unsigned long long xmx_buf_total_bytes(void)
{
	unsigned long long total = 0;
	for (int i = 0; i < MAX_BUF; i++)
		if (g.buf[i].used && !g.buf[i].internal)
			total += g.buf[i].bytes;
	return total;
}

int xmx_buf_zero(int id)
{
	struct buffer *b = slot(id);
	if (!b)
		return fail("bad buffer %d", id);
	if (b->host)
		memset(b->host, 0, b->bytes);
	b->gen++;
	/* Clear the device now, on the queue, before the commands that will use it: the scratch
	 * arena clears roles in the middle of a recording, so "now" is the only correct time. */
	return fill_device(id);
}

int xmx_buf_upload(int id, const void *src, unsigned long long off, unsigned long long bytes)
{
	struct buffer *b = slot(id);
	if (!b || off + bytes > b->bytes)
		return fail("upload out of range");
	if (!bytes)
		return 0;
	if (b->host)
		memcpy((char *)b->host + off, src, bytes);
	if (clEnqueueWriteBuffer(g.q, b->mem, CL_TRUE, (size_t)off, (size_t)bytes, src, 0, NULL, NULL)
	    != CL_SUCCESS)
		return fail("clEnqueueWriteBuffer: %d", -1);
	b->gen++;
	return 0;
}

int xmx_buf_download(int id, void *dst, unsigned long long off, unsigned long long bytes)
{
	struct buffer *b = slot(id);
	if (!b || off + bytes > b->bytes)
		return fail("download out of range");
	if (!bytes)
		return 0;
	if (b->host) {
		if (clEnqueueReadBuffer(g.q, b->mem, CL_TRUE, (size_t)off, (size_t)bytes,
					(char *)b->host + off, 0, NULL, NULL) != CL_SUCCESS)
			return fail("clEnqueueReadBuffer: %d", -1);
		memcpy(dst, (char *)b->host + off, bytes);
		return 0;
	}
	if (clEnqueueReadBuffer(g.q, b->mem, CL_TRUE, (size_t)off, (size_t)bytes, dst, 0, NULL, NULL)
	    != CL_SUCCESS)
		return fail("clEnqueueReadBuffer: %d", -1);
	return 0;
}

int xmx_buf_destroy(int id)
{
	struct buffer *b = slot(id);
	if (!b)
		return 0;
	if (b->mem)
		clReleaseMemObject(b->mem);
	if (b->packed)
		clReleaseMemObject(b->packed);
	free(b->host);
	memset(b, 0, sizeof *b);
	return 0;
}

/* ---- recording -------------------------------------------------------------------- */
int xmx_begin(void) { free_parts(g.cmd, g.ncmd); g.ncmd = 0; return 0; }
int xmx_abort(void) { free_parts(g.cmd, g.ncmd); g.ncmd = 0; return 0; }
int xmx_sync(int on) { (void)on; return 0; }

int xmx_rec_copy(int source, int target, unsigned long long bytes,
		 unsigned long long so, unsigned long long to)
{
	if (!slot(source) || !slot(target))
		return fail("copy: bad buffer");
	unsigned long long as = slot(source)->bytes, bs = slot(target)->bytes;
	if (!bytes || ((bytes | so | to) & 3) || so > as || bytes > as - so || to > bs || bytes > bs - to)
		return fail("copy range must be aligned and within both buffers");
	if (source == target && so < to + bytes && to < so + bytes)
		return fail("overlapping copy");
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *c = new_cmd(0);
	c->kind = K_COPY;
	c->a = source; c->c = target; c->bytes = bytes; c->so = so; c->to = to;
	c->stamp = PK_COPY * 32;
	return 0;
}

/* Every recorded command starts cleared. The list's slots are reused submit after
 * submit, and a recorder that sets only its own fields otherwise inherits the last
 * command's: gemm_residual ran with a stale ldc=128 at N=256, overlapping rows raced,
 * and the frame differed run to run from encoder L4 on (scripts/opencl-trace.sh). */
static struct cmd *new_cmd(int kind)
{
	struct cmd *c = g.cmd + g.ncmd;
	g.ncmd++;
	memset(c, 0, sizeof *c);
	c->kind = kind;
	c->d = -1;
	c->second = -1;
	return c;
}

/* A composed op's lowered commands hang off their K_COMPOSED command; they are freed when
 * the recording is submitted, aborted, or its graph destroyed. */
static void free_parts(struct cmd *list, int n)
{
	for (int i = 0; i < n; i++) {
		if (!list[i].parts)
			continue;
		free_parts(list[i].parts, list[i].nparts);
		free(list[i].parts);
		list[i].parts = NULL;
		list[i].nparts = 0;
	}
}

/* Fold the commands recorded since `base` into one composed op: the graph sees one, the
 * executor lowers it into all of them, so the recorded op stream stays the graph's. The
 * one recorded op carries one profile kind (`stamp`), so the per-pass table shows the
 * fused pass libxmx reports, not the several kernels this backend lowers it into. */
static int wrap_composed(int base, int stamp)
{
	int n = g.ncmd - base;
	if (n <= 0)
		return 0;
	if (n == 1) {                            /* nothing to fold: stamp the one command */
		g.cmd[base].stamp = stamp;
		return 0;
	}
	struct cmd *parts = malloc(sizeof(struct cmd) * (size_t)n);
	if (!parts)
		return fail("compose out of memory");
	memcpy(parts, &g.cmd[base], sizeof(struct cmd) * (size_t)n);
	g.ncmd = base;
	struct cmd *c = new_cmd(K_COMPOSED);
	c->parts = parts;
	c->nparts = n;
	c->stamp = stamp;
	return 0;
}

int xmx_rec_gemm(int a, int b, int c, unsigned M, unsigned N, unsigned K, unsigned batch,
		 unsigned sa, unsigned sb, unsigned sc, unsigned bt,
		 unsigned lda, unsigned ldb, unsigned ldc, unsigned oa, unsigned ob, unsigned oc)
{
	if (!slot(a) || !slot(b) || !slot(c))
		return fail("gemm: bad buffer");
	if (M % 8 || N % (8 * g.rn) || K % 16)
		return fail("gemm %ux%ux%u: shape not a multiple of the %dx%d tile", M, N, K,
			    8, 8 * g.rn);
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = new_cmd(0);
	cm->kind = K_GEMM;
	cm->a = a; cm->b = b; cm->c = c;
	cm->M = M; cm->N = N; cm->K = K; cm->batch = batch;
	cm->sa = sa; cm->sb = sb; cm->sc = sc; cm->bt = bt;
	cm->lda = lda; cm->ldb = ldb; cm->ldc = ldc;
	cm->oa = oa; cm->ob = ob; cm->oc = oc;
	cm->stamp = PK_GEMM * 32 + (int)(bt & 31u);
	return 0;
}

/* Pack B into the VNNI fragments gemm_blk wants, once per write. The packed copy lives on
 * the buffer and is remade only when the buffer's generation moves (a weight packs once;
 * an activation-B repacks whenever a command has written it) or its shape changes. */
static cl_mem ensure_packed(struct buffer *b, unsigned K, unsigned N, unsigned ldb,
			    unsigned sb, unsigned batch)
{
	unsigned long long want = (unsigned long long)batch * K * N * 2ull;
	if (!getenv("XMX_CL_BLK_NOCACHE") && b->packed && b->packed_gen == b->gen
	    && b->packed_K == K && b->packed_N == N && b->packed_stride == sb
	    && b->packed_batch == batch)
		return b->packed;
	if (b->packed && (b->packed_K != K || b->packed_N != N || b->packed_batch != batch)) {
		clReleaseMemObject(b->packed);
		b->packed = NULL;
	}
	if (!b->packed) {
		cl_int err;
		b->packed = clCreateBuffer(g.ctx, CL_MEM_READ_WRITE, want ? (size_t)want : 1, NULL, &err);
		if (!b->packed)
			return NULL;
	}
	cl_uint kk = K, nn = N, ll = ldb, ss = sb;
	int i = 0;
	clSetKernelArg(g.pack, i++, sizeof(cl_mem), &b->mem);
	clSetKernelArg(g.pack, i++, sizeof(cl_mem), &b->packed);
	clSetKernelArg(g.pack, i++, sizeof(cl_uint), &kk);
	clSetKernelArg(g.pack, i++, sizeof(cl_uint), &nn);
	clSetKernelArg(g.pack, i++, sizeof(cl_uint), &ll);
	clSetKernelArg(g.pack, i++, sizeof(cl_uint), &ss);
	size_t gsz = (K / 16u) * (N / 8u) * batch;
	if (clEnqueueNDRangeKernel(g.q, g.pack, 1, NULL, &gsz, NULL, 0, NULL, NULL) != CL_SUCCESS)
		return NULL;
	b->packed_gen = b->gen;
	b->packed_K = K;
	b->packed_N = N;
	b->packed_stride = sb;
	b->packed_batch = batch;
	return b->packed;
}

static int run_gemm(const struct cmd *c)
{
struct buffer *A = slot(c->a), *B = slot(c->b), *C = slot(c->c);
	/* One 8-row tile per workgroup, two tiles (RM=2) when M divides, as HEAD did. The
	 * run-to-run difference this used to be blamed for was uninitialised buffer memory,
	 * not the compiler (see xmx_buf_create_kind and notes/AUDIT-opencl-defects.md). */
	unsigned rm = (c->M % 16u) ? 1u : 2u;
	cl_kernel k = rm == 1u ? g.gemm_small : g.gemm;
	unsigned rn = g.rn;
	unsigned transposed = c->bt & 1u;
	unsigned epilogue = (c->bt >> 8) & 0xFu, narrow = (c->bt >> 12) & 1u;
	unsigned compact = (c->bt >> 16) & 1u;
	/* zero means dense, exactly as xmxres.gemm documents; resolved here so the kernel
	 * never has to. lda is A's row stride, ldb B's (K when B is stored transposed), ldc C's. */
	cl_uint lda = c->lda ? c->lda : c->K;
	cl_uint ldb = c->ldb ? c->ldb : (transposed ? c->K : c->N);
	cl_uint ldc = c->ldc ? c->ldc : c->N;
	cl_uint oa = c->oa, ob = c->ob, oc = c->oc;
	cl_uint sa = c->sa, sb = c->sb, sc = c->sc;
	cl_uint M = c->M, N = c->N, K = c->K;
	/* gemm_residual folds the skip in the epilogue: `d` is the skip, `second` the cosine
	 * (one value per channel). Both views of the skip share its mem - Sk half, Sf float. */
	cl_uint residual = (c->bt >> 17) & 1u;
	cl_uint skip_half = (c->bt >> 18) & 1u;
	struct buffer *skb = slot(c->d), *cob = slot(c->second);
	cl_mem skipmem = (residual && skb) ? skb->mem : A->mem;
	cl_mem cosmem = (residual && cob) ? cob->mem : A->mem;
	if (getenv("XMX_GEMM_DEBUG"))
		fprintf(stderr, "gemm M=%u N=%u K=%u batch=%u lda=%u ldb=%u ldc=%u oa=%u ob=%u oc=%u res=%u\n",
			M, N, K, c->batch, lda, ldb, ldc, oa, ob, oc, residual);
	/* The block-read path: B dense and untransposed, A dense with an even row stride, the
	 * offsets word-aligned (the block reads are 32-bit). A plain dense weight B packs once;
	 * anything else (a transposed B, a strided slice, an odd offset) stays on the scalar
	 * kernel, which reads B itself. */
	unsigned batch_n = c->batch ? c->batch : 1;
	cl_mem bmem = B->mem;
	if (g.gemm_blk && g.pack && !transposed && ob == 0 && lda == K && ldb == N
	    && !(oa & 1u) && !(sa & 1u) && (K % 16u) == 0 && (N % 8u) == 0
	    && c->bt == 0) {
		cl_mem packed = ensure_packed(B, K, N, ldb, sb, batch_n);
		if (packed) {
			k = g.gemm_blk;
			bmem = packed;
			rm = 1;
			rn = 1;
			if (getenv("XMX_GEMM_DEBUG"))
				fprintf(stderr, "  -> block-read GEMM, B packed\n");
		} else if (getenv("XMX_GEMM_DEBUG")) {
			fprintf(stderr, "  -> pack failed, scalar GEMM\n");
		}
	}
	int i = 0;
	clSetKernelArg(k, i++, sizeof(cl_mem), &A->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &bmem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &C->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &skipmem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &skipmem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &cosmem);
	clSetKernelArg(k, i++, sizeof(cl_uint), &M);
	clSetKernelArg(k, i++, sizeof(cl_uint), &N);
	clSetKernelArg(k, i++, sizeof(cl_uint), &K);
	clSetKernelArg(k, i++, sizeof(cl_uint), &lda);
	clSetKernelArg(k, i++, sizeof(cl_uint), &ldb);
	clSetKernelArg(k, i++, sizeof(cl_uint), &ldc);
	clSetKernelArg(k, i++, sizeof(cl_uint), &oa);
	clSetKernelArg(k, i++, sizeof(cl_uint), &ob);
	clSetKernelArg(k, i++, sizeof(cl_uint), &oc);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sa);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sb);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sc);
	clSetKernelArg(k, i++, sizeof(cl_uint), &transposed);
	clSetKernelArg(k, i++, sizeof(cl_uint), &epilogue);
	clSetKernelArg(k, i++, sizeof(cl_uint), &narrow);
	clSetKernelArg(k, i++, sizeof(cl_uint), &compact);
	clSetKernelArg(k, i++, sizeof(cl_uint), &residual);
	clSetKernelArg(k, i++, sizeof(cl_uint), &skip_half);
	cl_uint window_res = (c->bt >> 19) & 1u;
	cl_uint win_h = c->height, win_w = c->width, win_across = c->across, win_pad = c->pad;
	clSetKernelArg(k, i++, sizeof(cl_uint), &window_res);
	clSetKernelArg(k, i++, sizeof(cl_uint), &win_h);
	clSetKernelArg(k, i++, sizeof(cl_uint), &win_w);
	clSetKernelArg(k, i++, sizeof(cl_uint), &win_across);
	clSetKernelArg(k, i++, sizeof(cl_uint), &win_pad);
	unsigned batch = batch_n;
	size_t work[3] = { (size_t)N / (8 * rn) * 8, (size_t)M / (8 * rm), batch };
	size_t group[3] = { 8, 1, 1 };
	cl_event ev = NULL;
	cl_int err = clEnqueueNDRangeKernel(g.q, k, 3, NULL, work, group, 0, NULL,
					    profiling() ? &ev : NULL);
	if (err != CL_SUCCESS)
		return fail("gemm dispatch: %d", err);
	if (getenv("XMX_CL_SERIAL"))      /* diagnostic: one kernel in flight at a time */
		clFinish(g.q);
	trace_push(ev, 0);
	return 0;
}

/* The fused QKV-epilogue projection: A[M][K] x B[K][N] with N = 3C, a 32-column block per
 * workgroup (two subgroups of 8), then the normalise-and-publish epilogue in the kernel.
 * c->c/d/second are Q/K/V, c->n is the scale buffer, c->width/c->heads are tokens/heads. */
static int run_gemm_qkv(const struct cmd *c)
{
	struct buffer *A = slot(c->a), *B = slot(c->b);
	struct buffer *Q = slot(c->c), *Kb = slot(c->d), *V = slot(c->second);
	struct buffer *S = slot((int)c->n);
	if (!A || !B || !Q || !Kb || !V || !S)
		return fail("gemm_qkv: bad buffer");
	cl_uint M = c->M, N = c->N, K = c->K, lda = c->lda, ldb = c->ldb;
	cl_uint oa = c->oa, ob = c->ob, tokens = c->width, heads = c->heads, vit = c->flags;
	int i = 0;
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_mem), &A->mem);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_mem), &B->mem);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_mem), &Q->mem);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_mem), &Kb->mem);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_mem), &V->mem);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_mem), &S->mem);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &M);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &N);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &K);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &lda);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &ldb);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &oa);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &ob);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &tokens);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &heads);
	clSetKernelArg(g.gemm_qkv, i++, sizeof(cl_uint), &vit);
	size_t work[3] = { (size_t)(N / 32u) * 16u, (size_t)(M / 16u), 1u };
	size_t group[3] = { 16, 1, 1 };
	cl_event ev = NULL;
	cl_int err = clEnqueueNDRangeKernel(g.q, g.gemm_qkv, 3, NULL, work, group, 0, NULL,
					    profiling() ? &ev : NULL);
	if (err != CL_SUCCESS)
		return fail("gemm_qkv dispatch: %d", err);
	if (getenv("XMX_CL_SERIAL"))
		clFinish(g.q);
	trace_push(ev, 0);
	return 0;
}

/* The window attention finished on-chip: one workgroup a (window, head), Q/K/V in
 * (window, head, token, 32) order. c->c/d/second are Q/K/V, c->a is the optional bias,
 * c->b is the context (unmerged) output, c->M is batches (windows*heads), c->N is heads,
 * c->flags bit 0 selects the merged output at c->b. */
static int run_attn(const struct cmd *c)
{
	struct buffer *Q = slot(c->c), *Kb = slot(c->d), *V = slot(c->second);
	struct buffer *Out = slot(c->b);
	int merged = (int)(c->flags & 1u);
	if (!Q || !Kb || !V || !Out)
		return fail("attn: bad buffer");
	struct buffer *Bias = c->a >= 0 ? slot((int)c->a) : NULL;
	cl_uint batches = c->M, heads = c->N, merged_u = (cl_uint)merged;
	cl_uint bias_on = Bias ? 1u : 0u;
	cl_mem bias_mem = Bias ? Bias->mem : Out->mem;   /* unused when bias_on is 0 */
	int i = 0;
	clSetKernelArg(g.attn, i++, sizeof(cl_mem), &Q->mem);
	clSetKernelArg(g.attn, i++, sizeof(cl_mem), &Kb->mem);
	clSetKernelArg(g.attn, i++, sizeof(cl_mem), &V->mem);
	clSetKernelArg(g.attn, i++, sizeof(cl_mem), &bias_mem);
	clSetKernelArg(g.attn, i++, sizeof(cl_mem), &Out->mem);   /* the context or its dummy */
	clSetKernelArg(g.attn, i++, sizeof(cl_mem), &Out->mem);
	clSetKernelArg(g.attn, i++, sizeof(cl_uint), &batches);
	clSetKernelArg(g.attn, i++, sizeof(cl_uint), &heads);
	clSetKernelArg(g.attn, i++, sizeof(cl_uint), &merged_u);
	clSetKernelArg(g.attn, i++, sizeof(cl_uint), &bias_on);
	size_t work[1] = { (size_t)batches * 64u };
	size_t group[1] = { 64 };
	cl_event ev = NULL;
	cl_int err = clEnqueueNDRangeKernel(g.q, g.attn, 1, NULL, work, group, 0, NULL,
					    profiling() ? &ev : NULL);
	if (err != CL_SUCCESS)
		return fail("attn dispatch: %d", err);
	if (getenv("XMX_CL_SERIAL"))
		clFinish(g.q);
	trace_push(ev, 0);
	return 0;
}

static int run_unary(const struct cmd *c);
static int run_row(const struct cmd *c);

/* The model's own kinds; see xmxres.py (GRAPH, HOST_READ, HOST_WRITE). */
#define BUF_GRAPH 0
#define BUF_HOST_READ 1
#define BUF_HOST_WRITE 2

/* XMX_CL_TRACE=<path>: after every command, finish the queue and append a hash of each
 * buffer it wrote. Two runs' files diff to the first command whose output differs
 * (scripts/opencl-trace.sh). Diagnostic only: it serialises the queue. */
static unsigned long long hash_buffer(int id)
{
	struct buffer *b = (id >= 0 && id < MAX_BUF) ? &g.buf[id] : NULL;
	if (!b || !b->used || !b->bytes)
		return 0;
	unsigned char *data = malloc((size_t)b->bytes);
	if (!data)
		return 1;
	unsigned long long h = 1469598103934665603ull;
	if (clEnqueueReadBuffer(g.q, b->mem, CL_TRUE, 0, (size_t)b->bytes, data, 0, NULL, NULL)
	    == CL_SUCCESS)
		for (unsigned long long i = 0; i < b->bytes; i++)
			h = (h ^ data[i]) * 1099511628211ull;
	free(data);
	return h;
}

/* XMX_CL_DUMP_AT=<line> (1-based, counting trace lines): write that command's buffers a,
 * b, c, d, second raw to <XMX_CL_TRACE>.<line>.<role>, for frame_stages.py dumpdiff. */
static void dump_buffer(const char *base, long line, const char *role, int id)
{
	struct buffer *b = (id >= 0 && id < MAX_BUF) ? &g.buf[id] : NULL;
	if (!b || !b->used || !b->bytes)
		return;
	char path[1024];
	snprintf(path, sizeof path, "%s.%ld.%s", base, line, role);
	unsigned char *data = malloc((size_t)b->bytes);
	FILE *f = data ? fopen(path, "wb") : NULL;
	if (f && clEnqueueReadBuffer(g.q, b->mem, CL_TRUE, 0, (size_t)b->bytes, data, 0, NULL, NULL)
		 == CL_SUCCESS)
		fwrite(data, 1, (size_t)b->bytes, f);
	if (f)
		fclose(f);
	free(data);
}

static void trace_command(const struct cmd *c)
{
	static FILE *out;
	static int tried;
	static long line;
	static long index;
	static long dump_at;
	if (!tried) {
		const char *path = getenv("XMX_CL_TRACE");
		tried = 1;
		out = path && *path ? fopen(path, "a") : NULL;
		const char *at = getenv("XMX_CL_DUMP_AT");
		dump_at = at ? atol(at) : 0;
	}
	if (!out)
		return;
	clFinish(g.q);
	if (++line == dump_at) {
		const char *base = getenv("XMX_CL_TRACE");
		dump_buffer(base, line, "a", c->a);
		dump_buffer(base, line, "b", c->b);
		dump_buffer(base, line, "c", c->c);
		dump_buffer(base, line, "d", c->d);
		dump_buffer(base, line, "second", c->second);
		fprintf(out, "# dumped: lda=%u ldb=%u ldc=%u oa=%u ob=%u oc=%u sa=%u sb=%u sc=%u "
			"height=%u width=%u across=%u pad=%u\n", c->lda, c->ldb, c->ldc, c->oa, c->ob,
			c->oc, c->sa, c->sb, c->sc, c->height, c->width, c->across, c->pad);
	}
	fprintf(out, "%ld kind=%d a=%d b=%d c=%d d=%d second=%d M=%u N=%u K=%u batch=%u bt=%#x "
		"n=%u ch=%u k=%u rows=%u width=%u heads=%u flags=%#x out=%016llx",
		index++, c->kind, c->a, c->b, c->c, c->d, c->second, c->M, c->N, c->K, c->batch, c->bt,
		c->n, c->channels, c->k, c->rows, c->width, c->heads, c->flags, hash_buffer(c->c));
	if (c->kind == K_UNARY2)
		fprintf(out, " second=%016llx", hash_buffer(c->second));
	if (c->kind == K_ROW)
		fprintf(out, " d=%016llx", hash_buffer(c->d));
	fputc('\n', out);
	fflush(out);
}

/* One command on the queue. A K_COMPOSED op is the graph's single op, lowered here into
 * the several kernels it is made of (qkv, the fused projection, a window block); the
 * recorded op stream stays the graph's, so the pass counts match the fused backends. */
/* A command has written its target(s): move their generation so a VNNI-packed copy of one
 * is remade before the next GEMM that reads it (a write after a pack, later in the same
 * submit, is what made the single- and block-schedule frames differ). */
static void bump_written(const struct cmd *c)
{
	if (c->kind == K_ATTN) {              /* c->b is the context, c->a the merged output */
		if (c->b >= 0 && c->b < MAX_BUF && g.buf[c->b].used)
			g.buf[c->b].gen++;
		if ((c->flags & 1u) && c->a >= 0 && c->a < MAX_BUF && g.buf[c->a].used)
			g.buf[c->a].gen++;
		return;
	}
	if (c->c >= 0 && c->c < MAX_BUF && g.buf[c->c].used)
		g.buf[c->c].gen++;
	if (c->kind == K_UNARY2 && c->second >= 0 && c->second < MAX_BUF && g.buf[c->second].used)
		g.buf[c->second].gen++;
	else if (c->kind == K_ROW && c->d >= 0 && c->d < MAX_BUF && g.buf[c->d].used)
		g.buf[c->d].gen++;
	if (c->kind == K_GEMM_QKV) {          /* Q, K and V are all written */
		if (c->d >= 0 && c->d < MAX_BUF && g.buf[c->d].used)
			g.buf[c->d].gen++;
		if (c->second >= 0 && c->second < MAX_BUF && g.buf[c->second].used)
			g.buf[c->second].gen++;
	}
}

static int exec_one(const struct cmd *c)
{
	if (c->kind == K_GEMM) {
		if (run_gemm(c))
			return -1;
		bump_written(c);
		return 0;
	}
	if (c->kind == K_GEMM_QKV) {
		if (run_gemm_qkv(c))
			return -1;
		bump_written(c);
		return 0;
	}
	if (c->kind == K_ATTN) {
		if (run_attn(c))
			return -1;
		bump_written(c);
		return 0;
	}
	if (c->kind == K_UNARY || c->kind == K_UNARY2) {
		if (run_unary(c))
			return -1;
		bump_written(c);
		return 0;
	}
	if (c->kind == K_ROW) {
		if (run_row(c))
			return -1;
		bump_written(c);
		return 0;
	}
	if (c->kind == K_COMPOSED) {
		for (int j = 0; j < c->nparts; j++)
			if (exec_one(&c->parts[j]))
				return -1;
		return 0;
	}
	if (c->kind == K_COPY) {
		struct buffer *s = slot(c->a), *t = slot(c->c);
		/* always on the device, in queue order: the device copy is the current one once
		 * a kernel has written it, and the shadows sync at the ends of submit */
		cl_event ev = NULL;
		if (clEnqueueCopyBuffer(g.q, s->mem, t->mem, (size_t)c->so, (size_t)c->to,
					(size_t)c->bytes, 0, NULL, profiling() ? &ev : NULL) != CL_SUCCESS)
			return fail("clEnqueueCopyBuffer failed");
		trace_push(ev, 4);
		t->gen++;
		return 0;
	}
	return fail("unknown command kind %d", c->kind);
}

static void trace_command(const struct cmd *c);

static int exec_traced(const struct cmd *c)
{
	g_top_stamp = c->stamp;
	g_group++;
	if (c->kind == K_COMPOSED) {
		for (int j = 0; j < c->nparts; j++) {
			if (exec_one(&c->parts[j]))
				return -1;
			trace_command(&c->parts[j]);
		}
		return 0;
	}
	if (exec_one(c))
		return -1;
	trace_command(c);
	return 0;
}

/* Bind the pull set: every buffer a command (or a composed op's parts) writes. */
static void collect_pull(const struct cmd *c, unsigned char *pull)
{
	if (c->kind == K_COMPOSED) {
		for (int j = 0; j < c->nparts; j++)
			collect_pull(&c->parts[j], pull);
		return;
	}
	if (c->kind == K_ATTN) {              /* c->b is the context, c->a the merged output */
		if (c->b >= 0 && c->b < MAX_BUF) pull[c->b] = 1;
		if ((c->flags & 1u) && c->a >= 0 && c->a < MAX_BUF) pull[c->a] = 1;
		return;
	}
	if (c->c >= 0 && c->c < MAX_BUF) pull[c->c] = 1;
	if (c->kind == K_UNARY2 && c->second >= 0 && c->second < MAX_BUF) pull[c->second] = 1;
	if (c->kind == K_ROW && c->d >= 0 && c->d < MAX_BUF) pull[c->d] = 1;
	if (c->kind == K_GEMM_QKV) {          /* Q, K and V are all written */
		if (c->d >= 0 && c->d < MAX_BUF) pull[c->d] = 1;
		if (c->second >= 0 && c->second < MAX_BUF) pull[c->second] = 1;
	}
}

static int execute(const struct cmd *list, int n)
{
	trace_reset();
	/* Which buffers this submit writes, and what must be pushed up. A written buffer's
	 * generation moves, so its VNNI-packed copy is remade before the GEMM that reads it
	 * (packing is enqueued in queue order, after the write). Host-write buffers are the
	 * host's and always go up first. */
	unsigned char *pull = calloc(MAX_BUF, 1);
	if (!pull)
		return fail("pull map allocation");
	for (int i = 0; i < n; i++)
		collect_pull(&list[i], pull);
	for (int i = 0; i < MAX_BUF; i++) {
		struct buffer *b = &g.buf[i];
		if (b->used && b->kind == BUF_HOST_WRITE && b->host && b->bytes)
			b->gen++;      /* the host's content is new this frame */
	}
	/* Host-writable buffers the model filled through the pointer this frame: up first. */
	for (int i = 0; i < MAX_BUF; i++) {
		struct buffer *b = &g.buf[i];
		if (b->used && b->host && b->kind == BUF_HOST_WRITE && b->bytes
		    && clEnqueueWriteBuffer(g.q, b->mem, CL_TRUE, 0, (size_t)b->bytes, b->host, 0, NULL, NULL)
			       != CL_SUCCESS) {
			free(pull);
			return fail("push host-write buffer: %d", -1);
		}
	}
	for (int i = 0; i < n; i++) {
		if (exec_traced(&list[i])) {
			free(pull);
			return -1;
		}
	}
	clFinish(g.q);
	trace_sum();
	/* A host-write buffer a pass wrote is pushed up again at the next submit; pull it
	 * first, or that push puts back what the host last wrote and undoes the pass - which
	 * is what broke one-submit-a-block execution (test_frame_determinism.py). On Vulkan
	 * the memory is shared and this is free. ponytail: whole buffers, not ranges. */
	for (int i = 0; i < MAX_BUF; i++) {
		struct buffer *b = &g.buf[i];
		if (pull[i] && b->used && b->host && b->kind == BUF_HOST_WRITE && b->bytes
		    && clEnqueueReadBuffer(g.q, b->mem, CL_TRUE, 0, (size_t)b->bytes, b->host, 0, NULL, NULL)
			       != CL_SUCCESS) {
			free(pull);
			return fail("pull host-write buffer: %d", -1);
		}
	}
	free(pull);
	/* Device-written buffers the host will read back through the pointer: down last. */
	for (int i = 0; i < MAX_BUF; i++) {
		struct buffer *b = &g.buf[i];
		if (b->used && b->host && b->kind == BUF_HOST_READ && b->bytes
		    && clEnqueueReadBuffer(g.q, b->mem, CL_TRUE, 0, (size_t)b->bytes, b->host, 0, NULL, NULL)
			       != CL_SUCCESS)
			return fail("pull host-read buffer: %d", -1);
	}
	return n;
}

int xmx_submit(void)
{
	int n = g.ncmd;
	int r = execute(g.cmd, n);
	g.ncmd = 0;
	free_parts(g.cmd, n);
	return r < 0 ? r : n;
}

/* ---- graphs (a captured command list is just the recorded list) ------------------- */
int xmx_graph_capture(void)
{
	int id;
	for (id = 0; id < MAX_GRAPHS && g.graph[id].live; id++)
		;
	if (id == MAX_GRAPHS)
		return fail("out of graph slots");
	struct cmd *copy = malloc(sizeof(struct cmd) * (size_t)(g.ncmd ? g.ncmd : 1));
	if (!copy)
		return fail("graph capture out of memory");
	memcpy(copy, g.cmd, sizeof(struct cmd) * (size_t)g.ncmd);
	/* the graph now owns the composed ops' lowered commands: hand the pointers over */
	for (int i = 0; i < g.ncmd; i++)
		g.cmd[i].parts = NULL;
	g.graph[id].live = 1;
	g.graph[id].list = copy;
	g.graph[id].n = g.ncmd;
	g.ncmd = 0;
	return id;
}
int xmx_graph_run(int id)
{
	if (id < 0 || id >= MAX_GRAPHS || !g.graph[id].live)
		return fail("graph is not live");
	return execute(g.graph[id].list, g.graph[id].n);
}
int xmx_graph_destroy(int id)
{
	if (id < 0 || id >= MAX_GRAPHS || !g.graph[id].live)
		return 0;
	free_parts(g.graph[id].list, g.graph[id].n);
	free(g.graph[id].list);
	g.graph[id].live = 0;
	g.graph[id].list = NULL;
	g.graph[id].n = 0;
	collect_scratch();      /* scratch only this graph named is free now */
	return 0;
}

/* ---- plain GEMM path (gemm_runner) ------------------------------------------------ */
int xmx_init(const char *spv) { (void)spv; return g.open ? 0 : -1; }
int xmx_reserve(unsigned M, unsigned N, unsigned K, void **pa, void **pb, void **pc);
/* `xmx_gemm`/`xmx_gemm_batched` push the reserved shadows, dispatch once, read back. */
static int reserve_side(int *id, void **host, unsigned long long *sz, unsigned long long bytes)
{
	if (bytes <= *sz && *id >= 0 && slot(*id))
		return 0;
	if (*id >= 0) {
		xmx_buf_destroy(*id);
		*id = -1;
	}
	free(*host);
	*host = NULL;
	*sz = 0;
	if (!bytes)
		return 0;
	*host = calloc(1, (size_t)bytes);
	if (!*host)
		return fail("reserve out of memory");
	*id = xmx_buf_create_kind(bytes, 0);
	if (*id < 0) {
		free(*host);
		*host = NULL;
		return -1;
	}
	g.buf[*id].internal = 1;
	*sz = bytes;
	return 0;
}

int xmx_reserve_bytes(unsigned long long a, unsigned long long b, unsigned long long c,
		      void **pa, void **pb, void **pc)
{
	if (!g.open)
		return fail("device not open");
	if (reserve_side(&g_ria, &g_ra, &g_rasz, a) || reserve_side(&g_rib, &g_rb, &g_rbsz, b)
	    || reserve_side(&g_ric, &g_rc, &g_rcsz, c))
		return -1;
	if (pa)
		*pa = g_ra;
	if (pb)
		*pb = g_rb;
	if (pc)
		*pc = g_rc;
	return 0;
}

/* The typed reserve the `gemm_mapped` zero-copy path uses: half operands, float32 out. */
int xmx_reserve(unsigned M, unsigned N, unsigned K, void **pa, void **pb, void **pc)
{
	return xmx_reserve_bytes((unsigned long long)M * K * 2u, (unsigned long long)K * N * 2u,
				 (unsigned long long)M * N * 4u, pa, pb, pc);
}

static int push_side(int id, const void *src, unsigned long long bytes)
{
	if (clEnqueueWriteBuffer(g.q, slot(id)->mem, CL_TRUE, 0, (size_t)bytes, src, 0, NULL, NULL)
	    != CL_SUCCESS)
		return fail("descriptor gemm operand upload");
	return 0;
}

/* One recorded GEMM, then the queue drained and `c` read back. a, b or c may be NULL, in
 * which case the reserved shadow stands in for it (the `gemm_mapped` zero-copy path). */
static int descriptor_gemm(unsigned M, unsigned N, unsigned K, const void *a, const void *b,
			   void *c, unsigned batch, unsigned sa, unsigned sb, unsigned sc,
			   unsigned bt, unsigned iters)
{
	if (M % 8u || N % (unsigned)(8 * g.rn) || K % 16u)
		return fail("descriptor GEMM shape not a multiple of the tile");
	unsigned long long need_a = (unsigned long long)M * K * 2u;
	unsigned long long need_b = (unsigned long long)K * N * 2u;
	unsigned long long need_c = (unsigned long long)M * N * 4u;
	if (batch > 1u) {
		need_a = (unsigned long long)batch * sa * 2u;
		need_b = (unsigned long long)batch * sb * 2u;
		need_c = (unsigned long long)batch * sc * 4u;
	}
	if (reserve_side(&g_ria, &g_ra, &g_rasz, need_a)
	    || reserve_side(&g_rib, &g_rb, &g_rbsz, need_b)
	    || reserve_side(&g_ric, &g_rc, &g_rcsz, need_c))
		return -1;
	if (push_side(g_ria, a ? a : g_ra, need_a) || push_side(g_rib, b ? b : g_rb, need_b))
		return -1;
	struct cmd cm;
	memset(&cm, 0, sizeof cm);
	cm.kind = K_GEMM;
	cm.a = g_ria; cm.b = g_rib; cm.c = g_ric;
	cm.M = M; cm.N = N; cm.K = K; cm.batch = batch;
	cm.sa = sa; cm.sb = sb; cm.sc = sc; cm.bt = bt;
	for (unsigned i = 0; i < (iters ? iters : 1u); i++)
		if (run_gemm(&cm))
			return -1;
	if (clEnqueueReadBuffer(g.q, slot(g_ric)->mem, CL_TRUE, 0, (size_t)need_c,
				c ? c : g_rc, 0, NULL, NULL) != CL_SUCCESS)
		return fail("descriptor gemm result readback");
	return 0;
}

int xmx_gemm(unsigned M, unsigned N, unsigned K, const void *a, const void *b, void *c,
	     unsigned iters)
{
	if (!g.open)
		return fail("device not open");
	return descriptor_gemm(M, N, K, a, b, c, 1u, M * K, K * N, M * N, 0u, iters);
}
int xmx_init_batched(const char *spv) { (void)spv; return g.open ? 0 : -1; }
int xmx_gemm_batched(unsigned M, unsigned N, unsigned K, unsigned batch,
		     unsigned sa, unsigned sb, unsigned sc, unsigned bt)
{
	if (!g.open)
		return fail("device not open");
	return descriptor_gemm(M, N, K, NULL, NULL, NULL, batch, sa, sb, sc, bt, 1u);
}

/* ---- resident / fused families: one at a time, each against its Vulkan test -------- */
#define TODO(name) int name(void) { return fail(#name " not implemented yet"); }
int xmx_res_init(const char *a, const char *b, const char *c, const char *d, const char *e, const char *f)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f; return g.open ? 0 : -1; }
int xmx_staged32_init(const char *a, const char *b) { (void)a; (void)b; return 0; }
int xmx_staged32(unsigned on) { (void)on; return 0; }
unsigned xmx_staged32_calls(void) { return 0; }
int xmx_staged_partial(unsigned on) { (void)on; return 0; }
int xmx_rows_init(const char *p) { (void)p; return 0; }
int xmx_window_init(const char *p, unsigned m) { (void)p; (void)m; return 0; }
int xmx_ffn_init(const char *p) { (void)p; return 0; }
int xmx_window_block_init(const char *p) { (void)p; return 0; }
int xmx_global_attention_init(const char *p) { (void)p; return 0; }
int xmx_int8_init(const char *p) { (void)p; return 0; }

int xmx_rec_gemm_residual(int a, int b, int c, int skip, int cosine, unsigned M, unsigned N, unsigned K, unsigned flags)
{
	if (!slot(a) || !slot(b) || !slot(c) || !slot(skip) || !slot(cosine))
		return fail("gemm_residual: bad buffer");
	if (M % 8 || N % (8 * g.rn) || K % 16)
		return fail("gemm_residual %ux%ux%u: bad shape", M, N, K);
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = new_cmd(0);
	cm->kind = K_GEMM;
	cm->a = a; cm->b = b; cm->c = c; cm->d = skip; cm->second = cosine;
	cm->M = M; cm->N = N; cm->K = K; cm->batch = 1;
	cm->sa = M * K; cm->sb = K * N; cm->sc = M * N;
	cm->bt = flags | 0x20000u;
	cm->stamp = PK_GEMM * 32 + (int)(cm->bt & 31u);
	return 0;
}
int xmx_rec_gemm_window_residual(int a, int b, int c, int skip, int cosine, unsigned M, unsigned N, unsigned K, unsigned flags, unsigned h, unsigned w, unsigned ac, unsigned pad)
{
	if (!slot(a) || !slot(b) || !slot(c) || !slot(skip) || !slot(cosine) || !h || !w || !ac)
		return fail("gemm_window_residual: bad buffer");
	if (M % 8 || N % (8 * g.rn) || K % 16)
		return fail("gemm_window_residual %ux%ux%u: bad shape", M, N, K);
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = new_cmd(0);
	cm->kind = K_GEMM;
	cm->a = a; cm->b = b; cm->c = c; cm->d = skip; cm->second = cosine;
	cm->M = M; cm->N = N; cm->K = K; cm->batch = 1;
	cm->sa = M * K; cm->sb = K * N; cm->sc = M * N;
	cm->height = h; cm->width = w; cm->across = ac; cm->pad = pad;
	cm->bt = flags | 0x20000u | 0x80000u;
	cm->stamp = PK_GEMM * 32 + (int)(cm->bt & 31u);
	return 0;
}

/* Scratch for the composed fused passes. In-order queue and passes recorded in sequence,
 * so one buffer per role is enough. A role that must grow gets a new buffer; the old one
 * is freed unless the open recording or a live graph still names it, in which case it is
 * kept (and becomes collectable once that graph is destroyed: `collect_scratch`). */
enum { SCR_HIDDEN, SCR_MADE32, SCR_MADE16, SCR_PROJ, SCR_WIN16, SCR_SCORES, SCR_PROBS,
       SCR_CTX, SCR_RECIP, SCR_Q, SCR_K, SCR_V, SCR_ATT, SCR_OUT16, SCR_ROLES };
static int g_scratch[SCR_ROLES] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

static int names_buffer(const struct cmd *list, int n, int id)
{
	for (int i = 0; i < n; i++) {
		const struct cmd *c = &list[i];
		if (c->a == id || c->b == id || c->c == id || c->d == id || c->second == id)
			return 1;
		if (c->kind == K_COMPOSED && c->parts && names_buffer(c->parts, c->nparts, id))
			return 1;
	}
	return 0;
}

static int scratch_referenced(int id)
{
	if (names_buffer(g.cmd, g.ncmd, id))
		return 1;
	for (int i = 0; i < MAX_GRAPHS; i++)
		if (g.graph[i].live && names_buffer(g.graph[i].list, g.graph[i].n, id))
			return 1;
	return 0;
}

/* Free superseded scratch nothing names any more. */
static void collect_scratch(void)
{
	for (int i = 0; i < MAX_BUF; i++) {
		if (!g.buf[i].used || !g.buf[i].internal)
			continue;
		int current = 0;
		for (int r = 0; r < SCR_ROLES; r++)
			current |= g_scratch[r] == i;
		if (!current && !scratch_referenced(i))
			xmx_buf_destroy(i);
	}
}

static void scratch_reset(void)
{
	for (int r = 0; r < SCR_ROLES; r++)
		g_scratch[r] = -1;
}

static int scratch(int role, unsigned long long bytes)
{
	int id = g_scratch[role];
	if (id >= 0 && slot(id) && slot(id)->bytes >= bytes)
		return id;
	int grown = xmx_buf_create_kind(bytes, 0);
	if (grown < 0)
		return grown;
	g.buf[grown].internal = 1;
	g_scratch[role] = grown;
	collect_scratch();
	return grown;
}
int xmx_rec_unary(unsigned kind, int a, int b, int c, int d, unsigned n, unsigned channels,
		  float p0, unsigned batch, unsigned sa, unsigned sb, unsigned sc, unsigned k);
int xmx_rec_row(unsigned kind, int a, int b, int c, int d, unsigned rows, unsigned width,
		unsigned heads, unsigned scaled, unsigned stride, float cap);

/* The fused projection is used when the kernel built and the shape is tile-aligned: an
 * M multiple of the 16-row block (two subgroups, RM=2) and a column block of 32 that
 * divides 3C - which 3*heads*32 always does. NR_QKV_EPILOGUE=0 restores the four passes. */
static int qkv_fuse_ok(unsigned M, unsigned channels)
{
	if (!g.gemm_qkv)
		return 0;
	const char *env = getenv("NR_QKV_EPILOGUE");
	if (env && env[0] == '0' && env[1] == '\0')
		return 0;
	return (M % 16u) == 0u && (channels % 16u) == 0u;
}

/* The QKV projection. Fused (P3) the GEMM's epilogue normalises Q and K and publishes V
 * in the kernel, so the float32 projection never goes to memory; otherwise it is the four
 * passes it replaces - the GEMM into float32, cosine_publish for Q (with its head's scale)
 * and K, split_heads with the E4M3 publish for V. Both write the same bytes (test_gemm_qkv
 * compares the two paths directly, and the fused one against the reference). */
int xmx_rec_gemm_qkv(int a, int weight, int q, int k, int v, int scale, unsigned M,
		     unsigned channels, unsigned heads, unsigned tokens, unsigned vit)
{
	if (!heads || !tokens || channels != heads * 32u || M % tokens)
		return fail("invalid QKV projection");
	if (!slot(a) || !slot(weight) || !slot(q) || !slot(k) || !slot(v) || !slot(scale))
		return fail("QKV projection operand is not a live buffer");
	unsigned c3 = 3u * channels;
	if (qkv_fuse_ok(M, channels)) {
		if (g.ncmd >= MAX_CMD)
			return fail("command list full");
		struct cmd *cm = new_cmd(K_GEMM_QKV);
		cm->a = a; cm->b = weight; cm->c = q; cm->d = k; cm->second = v;
		cm->M = M; cm->N = c3; cm->K = channels;
		cm->lda = channels; cm->ldb = c3; cm->oa = 0; cm->ob = 0;
		cm->width = tokens; cm->heads = heads; cm->n = (unsigned)scale; cm->flags = vit;
		cm->stamp = PK(PK_GEMM, 0);
		return 0;
	}
	int proj = scratch(SCR_PROJ, (unsigned long long)M * c3 * 4u);
	if (proj < 0)
		return -1;
	int base = g.ncmd;
	if (xmx_rec_gemm(a, weight, proj, M, c3, channels, 1, M * channels, channels * c3, M * c3,
			 0, 0, 0, 0, 0, 0, 0))
		return -1;
	unsigned row = 0x1000u | 0x20000u | (vit ? 0x80000u : 0u);  /* COSINE_PUBLISH, qkv gather */
	if (xmx_rec_row(row, proj, proj, q, scale, M * heads, tokens, heads, 1, 0, 0.0f)
	    || xmx_rec_row(row | (1u << 18), proj, proj, k, proj, M * heads, tokens, heads, 0, 0, 0.0f))
		return -1;
	/* SPLIT_HEADS (10), part 2, E4M3 publish into half */
	if (xmx_rec_unary(10u | (1u << 8) | 0x1000u, proj, proj, v, proj, M * channels, channels,
			  1.0f, heads, tokens, 2u, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 0));
}

/* The window-gathered projection: partition the image (float32 or half; E4M3 published on
 * the way in when asked) into a half buffer, then the projection on it. */
int xmx_rec_gemm_qkv_window(int image, int weight, int q, int k, int v, int scale, unsigned M,
			    unsigned channels, unsigned heads, unsigned tokens, unsigned width,
			    unsigned height, unsigned across, unsigned pad, unsigned image_mode)
{
	if (!width || !height || !across || tokens != 64u || image_mode > 3u)
		return fail("invalid window-gathered QKV projection");
	if (!slot(image))
		return fail("QKV projection image is not a live buffer");
	int win = scratch(SCR_WIN16, (unsigned long long)M * channels * 2u);
	if (win < 0)
		return -1;
	unsigned flags = 7u | 0x1000u | ((image_mode & 1u) ? 0x8000u : 0u)   /* PARTITION */
		       | ((image_mode & 2u) ? (1u << 8) : 0u);
	int base = g.ncmd;
	if (xmx_rec_unary(flags, image, image, win, image, M * channels, channels, 1.0f, 8u,
			  height, width, across, pad))
		return -1;
	if (xmx_rec_gemm_qkv(win, weight, q, k, v, scale, M, channels, heads, tokens, 0))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 0));
}
/* The float32 product, and the same product published to E4M3 as half: two GEMMs. The
 * epilogue publishes the float32 accumulator, which is the value the first one stores,
 * so the half copy equals an `e4m3_half` of `c` bit for bit. */
int xmx_rec_gemm_dual(int a, int b, int c, int half_copy, unsigned M, unsigned N, unsigned K)
{
	if (!slot(half_copy))
		return fail("invalid GEMM half copy");
	int base = g.ncmd;
	if (xmx_rec_gemm(a, b, c, M, N, K, 1, M * K, K * N, M * N, 0, 0, 0, 0, 0, 0, 0)
	    || xmx_rec_gemm(a, b, half_copy, M, N, K, 1, M * K, K * N, M * N,
			    (1u << 8) | 0x1000u, 0, 0, 0, 0, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 0));
}

static int rec_unary(unsigned kind, int a, int b, int c, int d, int second,
		     unsigned n, unsigned channels, float p0, unsigned batch,
		     unsigned sa, unsigned sb, unsigned sc, unsigned k)
{
	if (!slot(a) || !slot(c))
		return fail("unary: bad buffer");
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = new_cmd(0);
	cm->kind = second >= 0 ? K_UNARY2 : K_UNARY;
	cm->flags = kind;
	cm->a = a; cm->b = b; cm->c = c; cm->d = d; cm->second = second;
	cm->n = n; cm->channels = channels; cm->cap = p0;
	cm->batch = batch; cm->sa = sa; cm->sb = sb; cm->sc = sc; cm->k = k;
	cm->stamp = PK_UNARY * 32 + (int)(kind & 31u);
	return 0;
}

int xmx_rec_unary2(unsigned kind, int a, int b, int c, int d, int second, unsigned n, unsigned channels, float p0, unsigned batch, unsigned sa, unsigned sb, unsigned sc, unsigned k)
{
	return rec_unary(kind, a, b, c, d, second, n, channels, p0, batch, sa, sb, sc, k);
}
int xmx_rec_unary(unsigned kind, int a, int b, int c, int d, unsigned n, unsigned channels, float p0, unsigned batch, unsigned sa, unsigned sb, unsigned sc, unsigned k)
{
	return rec_unary(kind, a, b, c, d, -1, n, channels, p0, batch, sa, sb, sc, k);
}

static int run_unary(const struct cmd *c)
{
	struct buffer *A = slot(c->a), *B = slot(c->b), *C = slot(c->c), *D = slot(c->d);
	struct buffer *S = c->second >= 0 ? slot(c->second) : NULL;
	cl_kernel k = g.resident;
	if (!A || !C)
		return fail("unary: bad buffer at run time");
	if (!B)
		B = A;                  /* an unused operand still needs a valid binding */
	if (!D)
		D = A;
	cl_uint flags = c->flags, m = c->n, n = c->channels, kk = c->k;
	cl_uint batch = c->batch, sa = c->sa, sb = c->sb, sc = c->sc;
	cl_mem nullmem = NULL;
	float p0 = c->cap;
	int i = 0;
	clSetKernelArg(k, i++, sizeof(cl_uint), &flags);
	clSetKernelArg(k, i++, sizeof(cl_mem), &A->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &B->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &C->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &D->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), S ? &S->mem : &nullmem);
	clSetKernelArg(k, i++, sizeof(cl_uint), &m);
	clSetKernelArg(k, i++, sizeof(cl_uint), &n);
	clSetKernelArg(k, i++, sizeof(cl_uint), &kk);
	clSetKernelArg(k, i++, sizeof(cl_uint), &batch);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sa);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sb);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sc);
	clSetKernelArg(k, i++, sizeof(float), &p0);
	size_t local = 256, global = ((size_t)m + local - 1) / local * local;
	cl_event ev = NULL;
	cl_int err = clEnqueueNDRangeKernel(g.q, k, 1, NULL, &global, &local, 0, NULL,
					    profiling() ? &ev : NULL);
	if (err != CL_SUCCESS)
		return fail("unary dispatch: %d", err);
	trace_push(ev, c->kind == 2 ? 1 : 2);
	return 0;
}
int xmx_rec_row(unsigned kind, int a, int b, int c, int d, unsigned rows, unsigned width, unsigned heads, unsigned scaled, unsigned stride, float cap)
{
	if (!slot(a) || !slot(c))
		return fail("row: bad buffer");
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = new_cmd(0);
	cm->kind = K_ROW;
	cm->flags = kind;
	cm->a = a; cm->b = b; cm->c = c; cm->d = d;
	cm->M = rows; cm->N = width; cm->batch = heads; cm->k = scaled; cm->sa = stride;
	cm->cap = cap;
	cm->stamp = PK_ROW * 32 + (int)(kind & 31u);
	return 0;
}

static int run_row(const struct cmd *c)
{
	struct buffer *A = slot(c->a), *B = slot(c->b), *C = slot(c->c), *D = slot(c->d);
	cl_kernel k = g.row;
	cl_uint flags = c->flags, m = c->M, n = c->N, kk = c->k, batch = c->batch;
	cl_uint sa = c->sa, sb = c->sb, sc = c->sc;
	float p0 = c->cap;
	int i = 0;
	clSetKernelArg(k, i++, sizeof(cl_uint), &flags);
	clSetKernelArg(k, i++, sizeof(cl_mem), &A->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &B->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &C->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &D->mem);
	clSetKernelArg(k, i++, sizeof(cl_uint), &m);
	clSetKernelArg(k, i++, sizeof(cl_uint), &n);
	clSetKernelArg(k, i++, sizeof(cl_uint), &kk);
	clSetKernelArg(k, i++, sizeof(cl_uint), &batch);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sa);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sb);
	clSetKernelArg(k, i++, sizeof(cl_uint), &sc);
	clSetKernelArg(k, i++, sizeof(float), &p0);
	size_t local = 32, global = ((size_t)m + local - 1) / local * local;
	cl_event ev = NULL;
	cl_int err = clEnqueueNDRangeKernel(g.q, k, 1, NULL, &global, &local, 0, NULL,
					    profiling() ? &ev : NULL);
	if (err != CL_SUCCESS)
		return fail("row dispatch: %d", err);
	trace_push(ev, 3);
	return 0;
}
int xmx_rec_history(int history, int motion, int out, unsigned pixels, unsigned channels, unsigned height, unsigned width, unsigned absolute)
{ (void)history;(void)motion;(void)out;(void)pixels;(void)channels;(void)height;(void)width;(void)absolute; return fail("history: not implemented yet"); }
/* The joint Q/K/V preparation as the three passes it replaces: cosine_publish for Q with
 * its head's scale and for K, split_heads with the E4M3 publish for V. `source` is the
 * projection already in (window, token, 3, head, 32). Bit-identical to
 * xmx_rec_gemm_qkv's own three finishing passes (test_joint_qkv.py). */
int xmx_rec_qkv(int source, int q, int k, int v, int scale, unsigned rows, unsigned tokens,
		unsigned heads, unsigned vit)
{
	if (!heads || !tokens || rows % heads)
		return fail("invalid joint QKV extent");
	if (!slot(source) || !slot(q) || !slot(k) || !slot(v) || !slot(scale))
		return fail("joint QKV operand is not a live buffer");
	unsigned windows_tokens = rows / heads;
	unsigned channels = heads * 32u;
	unsigned row = 0x1000u | 0x20000u | (vit ? 0x80000u : 0u);  /* COSINE_PUBLISH, qkv gather */
	int base = g.ncmd;
	if (xmx_rec_row(row, source, source, q, scale, rows, tokens, heads, 1, 0, 0.0f)
	    || xmx_rec_row(row | (1u << 18), source, source, k, source, rows, tokens, heads, 0, 0,
			   0.0f))
		return -1;
	/* SPLIT_HEADS (10), part 2, E4M3 publish into half */
	if (xmx_rec_unary(10u | (1u << 8) | 0x1000u, source, source, v, source,
			  windows_tokens * channels, channels, 1.0f, heads, tokens, 2u, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 0));
}
/* Window attention as the passes it replaces (test_window_attention.py's reference): QK^T
 * into float32 scores, the bit-affine softmax (with the per-head bias) into half, PV into
 * the float32 context, and for `merged` the head merge with the E4M3 publish into half.
 * ponytail: scores, probabilities and context go through device memory; one kernel
 * keeping them on-chip is P3. */
int xmx_rec_window_attention(int q, int k, int v, int bias, int out, unsigned batches,
			     unsigned heads, unsigned merged)
{
	if (merged > 1u || !batches || !heads || batches % heads)
		return fail("invalid attention batch/head count");
	if (!slot(q) || !slot(k) || !slot(v) || !slot(out) || (bias >= 0 && !slot(bias)))
		return fail("window attention operand is not a live buffer");
	/* The fused path: QK^T, the bit-affine softmax and PV in one workgroup a (window,
	 * head), the scores and probabilities on-chip. Exact by construction against the
	 * decomposed passes below (test_window_attention.py). NR_WINDOW_ATTENTION_SPLIT=1
	 * forces the decomposed path for A/B. */
	if (g.attn && !getenv("NR_WINDOW_ATTENTION_SPLIT")) {
		if (g.ncmd >= MAX_CMD)
			return fail("command list full");
		struct cmd *cm = new_cmd(K_ATTN);
		cm->kind = K_ATTN;
		cm->c = q; cm->d = k; cm->second = v; cm->a = bias; cm->b = out;
		cm->M = batches; cm->N = heads; cm->flags = merged;
		cm->stamp = PK_ROW * 32 + 3;
		return 0;
	}
	int scores = scratch(SCR_SCORES, (unsigned long long)batches * 4096u * 4u);
	int probs = scores < 0 ? -1 : scratch(SCR_PROBS, (unsigned long long)batches * 4096u * 2u);
	int ctx = !merged ? out : probs < 0 ? -1
		: scratch(SCR_CTX, (unsigned long long)batches * 2048u * 4u);
	if (scores < 0 || probs < 0 || ctx < 0)
		return -1;
	int base = g.ncmd;
	if (xmx_rec_gemm(q, k, scores, 64, 64, 32, batches, 2048, 2048, 4096, 1u, 0, 0, 0, 0, 0, 0))
		return -1;
	unsigned soft = 1u | 0x1000u | (bias >= 0 ? 0x2000u : 0u);       /* SOFTMAX, half out */
	if (xmx_rec_row(soft, scores, bias >= 0 ? bias : scores, probs, scores, batches * 64u, 64,
			heads, 0, 0, 0.0f))
		return -1;
	if (xmx_rec_gemm(probs, v, ctx, 64, 32, 64, batches, 4096, 2048, 2048, 0, 0, 0, 0, 0, 0, 0))
		return -1;
	if (!merged)
		return wrap_composed(base, PK(PK_ROW, 3));
	/* MERGE_HEADS (11), E4M3 publish into half: (window, token, head, 32) */
	if (xmx_rec_unary(11u | (1u << 8) | 0x1000u, ctx, ctx, out, ctx, batches * 2048u,
			  heads * 32u, 1.0f, heads, 64, 0, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_ROW, 3));
}
/* The fused feed-forward as the two GEMMs it replaces: the expand with the gate and E4M3
 * publish into a half hidden buffer, then the projection (with the residual when there is
 * one). Bit-identical to the reference by construction. ponytail: the hidden layer goes
 * through device memory; fusing it into one kernel is P3 in docs/opencl-port.md. */
int xmx_rec_ffn(int a, int expand, int projection, int out, int skip, int cosine,
		unsigned M, unsigned cin, unsigned hidden, unsigned groups, unsigned flags)
{
	if (!cin || cin % 16u || !hidden || hidden % 32u || !M || M % 16u || !groups)
		return fail("fused feed-forward needs channels a multiple of 16, hidden of 32, 16-row blocks");
	if (flags & ~0x41f00u)
		return fail("fused feed-forward takes an epilogue, a half output and a half skip only");
	int residual = skip >= 0 || cosine >= 0;
	if (residual && (groups != 1u || cin != 32u || skip < 0 || cosine < 0))
		return fail("the fused residual is for one group of 32 channels, with skip and cosine");
	if (!residual && (flags & 0x40000u))
		return fail("a half skip without a residual");
	if (!slot(a) || !slot(expand) || !slot(projection) || !slot(out)
	    || (residual && (!slot(skip) || !slot(cosine))))
		return fail("fused feed-forward operand is not a live buffer");
	int hid = scratch(SCR_HIDDEN, (unsigned long long)M * groups * hidden * 2u);
	if (hid < 0)
		return -1;
	int base = g.ncmd;
	if (xmx_rec_gemm(a, expand, hid, M, hidden, cin, groups, 0, cin * hidden, hidden,
			 (3u << 8) | 0x1000u, 0, 0, groups * hidden, 0, 0, 0))
		return -1;
	if (residual) {
		if (xmx_rec_gemm_residual(hid, projection, out, skip, cosine, M, 32u, hidden, flags))
			return -1;
		return wrap_composed(base, PK(PK_GEMM, 31));
	}
	if (xmx_rec_gemm(hid, projection, out, M, 32u, hidden, groups, hidden, hidden * 32u, 32u,
			 flags & 0x1f00u, groups * hidden, 0, groups * 32u, 0, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 31));
}

/* Block 70: UPSAMPLE_MERGE (float32 and published half), then the feed-forward on them. */
int xmx_rec_ffn_merge(int source, int skip, int sincos, int expand, int projection, int out,
		      int cosine, unsigned height, unsigned width, unsigned source_width,
		      unsigned flags)
{
	unsigned M = height * width;
	if (!M || M % 16u || height % 2u || width % 2u || source_width < width / 2u)
		return fail("the merged feed-forward needs an even extent in 16-row blocks");
	if (flags & ~0x1f00u)
		return fail("the merged feed-forward takes an epilogue and a half output only");
	if (!slot(source) || !slot(skip) || !slot(sincos))
		return fail("merged feed-forward operand is not a live buffer");
	int m32 = scratch(SCR_MADE32, (unsigned long long)M * 32u * 4u);
	int m16 = scratch(SCR_MADE16, (unsigned long long)M * 32u * 2u);
	if (m32 < 0 || m16 < 0)
		return -1;
	int base = g.ncmd;
	if (xmx_rec_unary2(20u | 0x8000u | 0x10000u, source, skip, m32, sincos, m16, M * 32u, 32u,
			   1.0f, 0, width, source_width, 0, 0))
		return -1;
	if (xmx_rec_ffn(m16, expand, projection, out, m32, cosine, M, 32u, 128u, 1u, flags))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 31));
}

/* Block 0: the stem through gemm_dual, then the feed-forward on it. */
int xmx_rec_ffn_stem(int features, int adapter, int expand, int projection, int out,
		     int cosine, unsigned M, unsigned flags)
{
	if (!M || M % 16u)
		return fail("the stem feed-forward needs 16-row blocks");
	if (flags & ~0x1f00u)
		return fail("the stem feed-forward takes an epilogue and a half output only");
	int m32 = scratch(SCR_MADE32, (unsigned long long)M * 32u * 4u);
	int m16 = scratch(SCR_MADE16, (unsigned long long)M * 32u * 2u);
	if (m32 < 0 || m16 < 0)
		return -1;
	int base = g.ncmd;
	if (xmx_rec_gemm_dual(features, adapter, m32, m16, M, 32u, 16u))
		return -1;
	if (xmx_rec_ffn(m16, expand, projection, out, m32, cosine, M, 32u, 128u, 1u, flags))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 31));
}
/* Block 0's window residual with both readers: the window residual GEMM into float32,
 * then pool2_skip of it - what `gemm_residual(reverse)` followed by `pool2_skip` write.
 * ponytail: the float32 output goes through device memory; one epilogue kernel is P3. */
int xmx_rec_gemm_window_residual_pool(int a, int b, int skip_out, int skip, int cosine,
				      int pooled, unsigned M, unsigned N, unsigned K,
				      unsigned flags, unsigned height, unsigned width,
				      unsigned across, unsigned pad)
{
	if (!slot(a) || !slot(b) || !slot(skip_out) || !slot(skip) || !slot(cosine)
	    || !slot(pooled))
		return fail("window residual pool operand is not a live buffer");
	if (!height || !width || height % 2u || width % 2u || N != 32u)
		return fail("the pooled window residual needs even extents and 32 channels");
	unsigned pixels = height * width;
	int raw = scratch(SCR_PROJ, (unsigned long long)pixels * N * 4u);
	if (raw < 0)
		return -1;
	unsigned residual = flags & 0x40000u;   /* skip_half rides in the frame's own bit */
	int base = g.ncmd;
	if (xmx_rec_gemm_window_residual(a, b, raw, skip, cosine, M, N, K, residual,
					 height, width, across, pad))
		return -1;
	/* POOL2_SKIP (22), E4M3 publish into half `pooled`, e4m3_half into `skip_out` */
	if (xmx_rec_unary2(22u | (1u << 8) | 0x1000u, raw, raw, pooled, raw, skip_out,
			   pixels / 4u * N, N, 1.0f, 0, height, width, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_GEMM, 31));
}
/* A 32-channel window block's attention half as the three passes it replaces:
 * gemm_qkv_window, merged window attention, and the window residual - or, for the head,
 * the residual and the compact head GEMM. See xmxres.window_block. */
int xmx_rec_window_block(int image, int qkv, int projection, int target, int bias, int cosine,
			 int scale, int extra, unsigned windows, unsigned height, unsigned width,
			 unsigned across, unsigned pad, unsigned flags)
{
	if (!windows || !height || !width || !across)
		return fail("invalid window block geometry");
	if (!slot(image) || !slot(qkv) || !slot(projection) || !slot(target) || !slot(bias)
	    || !slot(cosine) || !slot(scale))
		return fail("window block operand is not a live buffer");
	unsigned image_half = (flags & 0x8000u) != 0u;
	unsigned narrow = (flags & 0x1000u) != 0u;
	unsigned epilogue = (flags >> 8) & 0xFu;
	unsigned pooled = (flags & 0x800000u) != 0u;
	unsigned head = (flags & 0x1000000u) != 0u;
	unsigned head16 = (flags & 0x2000000u) != 0u;
	if (pooled && head)
		return fail("the window block pools or takes the head, not both");
	if ((head || pooled) && extra < 0)
		return fail("the window block's pool or head output is missing");
	unsigned rows = windows * 64u;
	unsigned pixels = height * width;
	if (rows % 16u)
		return fail("the window block needs 16-row blocks");
	if (pooled && (height % 2u || width % 2u))
		return fail("the pooled window block needs even extents");
	int q = scratch(SCR_Q, (unsigned long long)rows * 32u * 2u);
	int k = q < 0 ? -1 : scratch(SCR_K, (unsigned long long)rows * 32u * 2u);
	int v = k < 0 ? -1 : scratch(SCR_V, (unsigned long long)rows * 32u * 2u);
	int att = v < 0 ? -1 : scratch(SCR_ATT, (unsigned long long)rows * 32u * 2u);
	if (att < 0)
		return -1;
	/* image_mode: bit 0 a half image, bit 1 the image published to E4M3 as it is loaded */
	unsigned image_mode = 2u | (image_half ? 1u : 0u);
	int base = g.ncmd;
	if (xmx_rec_gemm_qkv_window(image, qkv, q, k, v, scale, rows, 32u, 1u, 64u, width, height,
				    across, pad, image_mode)
	    || xmx_rec_window_attention(q, k, v, bias, att, windows, 1u, 1u))
		return -1;
	unsigned residual = (epilogue << 8) | (narrow ? 0x1000u : 0u) | (image_half ? 0x40000u : 0u);
	if (pooled) {
		if (xmx_rec_gemm_window_residual_pool(
			    att, projection, target, image, cosine, extra, rows, 32u, 32u,
			    (image_half ? 0x40000u : 0u), height, width, across, pad))
			return -1;
		return wrap_composed(base, PK(PK_ROW, 4));
	}
	if (!head) {
		if (xmx_rec_gemm_window_residual(att, projection, target, image, cosine, rows,
						 32u, 32u, residual, height, width, across, pad))
			return -1;
		return wrap_composed(base, PK(PK_ROW, 4));
	}
	/* Block 70: the residual into half `out16`, then the compact head GEMM on it. */
	int out16 = scratch(SCR_OUT16, (unsigned long long)pixels * 32u * 2u);
	if (out16 < 0)
		return -1;
	if (xmx_rec_gemm_window_residual(att, projection, out16, image, cosine, rows, 32u, 32u,
					 0x1000u | (image_half ? 0x40000u : 0u), height, width,
					 across, pad))
		return -1;
	/* N is always 16; compact stores the first four columns (ldc 4), the wide build all
	 * sixteen (ldc 16). See xmxres.gemm compact_output. */
	if (xmx_rec_gemm(out16, extra, target, pixels, 16u, 32u, 1, 0, 0, 0,
			 head16 ? 0u : (1u << 16), 0, 0, head16 ? 0u : 4u, 0, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_ROW, 4));
}
/* A bottleneck block's attention as the four passes it replaces (test_global_attention.py):
 * QK^T, the ViT's softmax with its reciprocals, PV, and the scaled head merge with the
 * E4M3 publish. ponytail: intermediates in device memory; one on-chip kernel is P3. */
int xmx_rec_global_attention(int q, int k, int v, int merged, unsigned rows, unsigned tokens,
			     unsigned heads)
{
	if (!rows || rows % 16u || !tokens || tokens > rows || !heads
	    || (rows + 63u) / 64u != (tokens + 63u) / 64u)
		return fail("global attention needs rows a multiple of 16 within the tokens' 64-block");
	if (!slot(q) || !slot(k) || !slot(v) || !slot(merged))
		return fail("global attention operand is not a live buffer");
	unsigned long long sq = (unsigned long long)heads * rows * rows;
	int scores = scratch(SCR_SCORES, sq * 4u);
	int probs = scores < 0 ? -1 : scratch(SCR_PROBS, sq * 2u);
	int recip = probs < 0 ? -1 : scratch(SCR_RECIP, (unsigned long long)heads * rows * 4u);
	int ctx = recip < 0 ? -1 : scratch(SCR_CTX, (unsigned long long)heads * rows * 32u * 4u);
	if (ctx < 0)
		return -1;
	int base = g.ncmd;
	if (xmx_rec_gemm(q, k, scores, rows, rows, 32, heads, rows * 32u, rows * 32u, rows * rows,
			    1u, 0, 0, 0, 0, 0, 0)
	    || xmx_rec_row(6u | 0x1000u, scores, scores, probs, recip, heads * rows, tokens, 0, 0,
			   rows, 0.0f)                                  /* VIT_SOFTMAX */
	    || xmx_rec_gemm(probs, v, ctx, rows, 32, rows, heads, rows * rows, rows * 32u,
			    rows * 32u, 0, 0, 0, 0, 0, 0, 0))
		return -1;
	/* MERGE_HEADS (11) scaled by the reciprocals, E4M3 publish into half */
	if (xmx_rec_unary(11u | 0x20000u | (1u << 8) | 0x1000u, ctx, ctx, merged, recip,
			  rows * heads * 32u, heads * 32u, 1.0f, heads, rows, 0, 0, 0))
		return -1;
	return wrap_composed(base, PK(PK_ROW, 5));
}
int xmx_rec_gemm_int8(int a, int b, int c, int a_scale, int b_scale, unsigned M, unsigned N, unsigned K, unsigned flags)
{ (void)a;(void)b;(void)c;(void)a_scale;(void)b_scale;(void)M;(void)N;(void)K;(void)flags; return fail("gemm_int8: not implemented yet"); }

/* ---- profiling -------------------------------------------------------------------- */
/* The table libxmx exposes: per-kind device milliseconds and hit counts read from the
 * cl_event timestamps, plus the flat per-dispatch list. A kind is family * 32 + subkind,
 * the same numbering libxmx uses, so frame_profile.py/iterate.sh read either backend. */
int xmx_profile(int on)
{
	if (on && !g.prof) {
		if (!g.q)
			return -1;
		if (!g.qprof) {
			clFinish(g.q);
			cl_int err = 0;
			cl_command_queue q = clCreateCommandQueue(g.ctx, g.dev, CL_QUEUE_PROFILING_ENABLE, &err);
			if (!q)
				return fail("profile: clCreateCommandQueue: %d", err);
			clReleaseCommandQueue(g.q);
			g.q = q;
			g.qprof = 1;
		}
		if (!g_trace) {
			g_trace = malloc(sizeof(*g_trace) * MAX_CMD);
			if (!g_trace)
				return fail("profile: out of memory");
		}
		g.prof = 1;
	} else if (!on) {
		g.prof = 0;
	}
	return 0;
}

void xmx_profile_reset(void)
{
	memset(prof_ms, 0, sizeof prof_ms);
	memset(prof_hits, 0, sizeof prof_hits);
	prof_each_n = 0;
}

double xmx_profile_ms(unsigned kind) { return kind < PROF_KINDS ? prof_ms[kind] : 0.0; }
unsigned xmx_profile_count(unsigned kind) { return kind < PROF_KINDS ? prof_hits[kind] : 0; }
unsigned xmx_profile_each_count(void) { return (unsigned)prof_each_n; }
double xmx_profile_each_ms(unsigned i) { return i < (unsigned)prof_each_n ? prof_each[i] : 0.0; }
unsigned xmx_profile_each_kind(unsigned i) { return i < (unsigned)prof_each_n ? prof_each_kind[i] : 0u; }