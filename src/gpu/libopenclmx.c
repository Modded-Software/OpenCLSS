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
"        for (unsigned c = 0; c < RN; c++) acc[r][c] = (float8)(0.0f);\n"
"    for (unsigned k = 0; k < K; k += 16) {\n"
"        int8 a[RM], b[RN];\n"
"#pragma unroll\n"
"        for (unsigned r = 0; r < RM; r++) {\n"
"            int8 t;\n"
"#pragma unroll\n"
"            for (unsigned i = 0; i < 8; i++)\n"
"                t[i] = as_int((half2)(A[(row + r * 8 + i) * lda + k + 2 * lane],\n"
"                                      A[(row + r * 8 + i) * lda + k + 2 * lane + 1]));\n"
"            a[r] = t;\n"
"        }\n"
"#pragma unroll\n"
"        for (unsigned c = 0; c < RN; c++) {\n"
"            int8 t;\n"
"#pragma unroll\n"
"            for (unsigned i = 0; i < 8; i++)\n"
"                t[i] = transposed\n"
"                    ? as_int((half2)(B[(col + c * 8 + lane) * ldb + k + 2 * i],\n"
"                                     B[(col + c * 8 + lane) * ldb + k + 2 * i + 1]))\n"
"                    : as_int((half2)(B[(k + 2 * i) * ldb + col + c * 8 + lane],\n"
"                                     B[(k + 2 * i + 1) * ldb + col + c * 8 + lane]));\n"
"            b[c] = t;\n"
"        }\n"
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
"        C[index] = publish_epi(epilogue, total * 0.25f);\n" \
"        Sh[taps[0]] = (half)e4m3(v0); Sh[taps[1]] = (half)e4m3(v1);\n" \
"        Sh[taps[2]] = (half)e4m3(v2); Sh[taps[3]] = (half)e4m3(v3);\n" \
"        return;\n" \
"    }\n" \
"    case UPSAMPLE_ADD: {\n" \
"        uint target = sa, source = sb; c = index % n; rest = index / n;\n" \
"        uint x = rest % target, y = rest / target;\n" \
"        uint at = ((y / 2u) * source + (x / 2u)) * n + c;\n" \
"        float scaled = (bhalf ? (float)Bh[index] : B[index]) * D[c];\n" \
"        float merged = (ahalf ? (float)Ah[at] : A[at]) + scaled;\n" \
"        C[index] = publish_epi(epilogue, merged);\n" \
"        if (S != (__global float *)0) Sh[index] = (half)e4m3(merged);\n" \
"        return;\n" \
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

/* ---- runtime state ---------------------------------------------------------------- */
#define MAX_BUF 8192
#define MAX_GRAPHS 64
#define MAX_CMD 65536
#define K_GEMM 0
#define K_COPY 1
#define K_UNARY 2
#define K_UNARY2 3
#define K_ROW 4

struct buffer {
	int used;
	void *host;                 /* non-NULL only for a mapped buffer: the memory `mem` wraps */
	cl_mem mem;
	unsigned long long bytes;
	int kind;                   /* GRAPH=0 (device-local, unmapped), HOST_READ/WRITE != 0 */
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
};

static struct {
	int open;
	cl_context ctx;
	cl_command_queue q;
	cl_device_id dev;
	cl_program prog;
	cl_kernel gemm;
	cl_kernel gemm_small;       /* RM=1: one 8-row tile, for M that is not a multiple of 16 */
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
	unsigned long long dev_ns;  /* summed device time of the last execute() */
	int dev_cmds;               /* dispatches profiled in the last execute() */
} g;

static struct { cl_event ev; int tag; } *g_trace;
static int g_ntrace;

static void trace_reset(void)
{
	if (!g.time)
		return;
	for (int i = 0; i < g_ntrace; i++)
		clReleaseEvent(g_trace[i].ev);
	g_ntrace = 0;
}

static void trace_push(cl_event ev, int tag)
{
	if (g.time && ev) {
		g_trace[g_ntrace].ev = ev;
		g_trace[g_ntrace].tag = tag;
		g_ntrace++;
	}
}

static void trace_sum(void)
{
	if (!g.time)
		return;
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
int xmx_portable_packed(void) { return 0; }
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
	if (clBuildProgram(prog, 1, &g.dev, opts, NULL, NULL) != CL_SUCCESS) {
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
	g.q = clCreateCommandQueue(g.ctx, g.dev, g.time ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
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
		g.gemm = build_kernel(src, NULL, "gemm");
		g.gemm_small = build_kernel(src, "-D RM=1", "gemm");
		free(src);
	}
	if (!g.gemm)
		return -1;
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

void xmx_close(void)
{
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
		g.total += bytes;
		if (kind != 0 && fill_device(i)) {
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
unsigned long long xmx_buf_total_bytes(void) { return g.total; }

int xmx_buf_zero(int id)
{
	struct buffer *b = slot(id);
	if (!b)
		return fail("bad buffer %d", id);
	if (b->host)
		memset(b->host, 0, b->bytes);
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
	free(b->host);
	memset(b, 0, sizeof *b);
	return 0;
}

/* ---- recording -------------------------------------------------------------------- */
int xmx_begin(void) { g.ncmd = 0; return 0; }
int xmx_abort(void) { g.ncmd = 0; return 0; }
int xmx_sync(int on) { (void)on; return 0; }

int xmx_rec_copy(int source, int target, unsigned long long bytes,
		 unsigned long long so, unsigned long long to)
{
	if (!slot(source) || !slot(target))
		return fail("copy: bad buffer");
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *c = &g.cmd[g.ncmd++];
	c->kind = K_COPY;
	c->a = source; c->c = target; c->bytes = bytes; c->so = so; c->to = to;
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
	struct cmd *cm = &g.cmd[g.ncmd++];
	cm->kind = K_GEMM;
	cm->a = a; cm->b = b; cm->c = c;
	cm->M = M; cm->N = N; cm->K = K; cm->batch = batch;
	cm->sa = sa; cm->sb = sb; cm->sc = sc; cm->bt = bt;
	cm->lda = lda; cm->ldb = ldb; cm->ldc = ldc;
	cm->oa = oa; cm->ob = ob; cm->oc = oc;
	return 0;
}

static int run_gemm(const struct cmd *c)
{
	struct buffer *A = slot(c->a), *B = slot(c->b), *C = slot(c->c);
	/* M that is not a multiple of 16 falls back to the one-tile kernel; every other
	 * dimension is already constrained at record time, so no masking is needed. */
	unsigned rm = (c->M % 16u) ? 1u : 2u;
	cl_kernel k = rm == 1u ? g.gemm_small : g.gemm;
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
	int i = 0;
	clSetKernelArg(k, i++, sizeof(cl_mem), &A->mem);
	clSetKernelArg(k, i++, sizeof(cl_mem), &B->mem);
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
	unsigned batch = c->batch ? c->batch : 1;
	size_t work[3] = { (size_t)N / (8 * g.rn) * 8, (size_t)M / (8 * rm), batch };
	size_t group[3] = { 8, 1, 1 };
	cl_event ev = NULL;
	cl_int err = clEnqueueNDRangeKernel(g.q, k, 3, NULL, work, group, 0, NULL,
					    g.time ? &ev : NULL);
	if (err != CL_SUCCESS)
		return fail("gemm dispatch: %d", err);
	trace_push(ev, 0);
	return 0;
}

static int run_unary(const struct cmd *c);
static int run_row(const struct cmd *c);

/* The model's own kinds; see xmxres.py (GRAPH, HOST_READ, HOST_WRITE). */
#define BUF_GRAPH 0
#define BUF_HOST_READ 1
#define BUF_HOST_WRITE 2

static int execute(const struct cmd *list, int n)
{
	trace_reset();
	/* Host-writable buffers the model filled through the pointer this frame: up first. */
	for (int i = 0; i < MAX_BUF; i++) {
		struct buffer *b = &g.buf[i];
		if (b->used && b->host && b->kind == BUF_HOST_WRITE && b->bytes
		    && clEnqueueWriteBuffer(g.q, b->mem, CL_TRUE, 0, (size_t)b->bytes, b->host, 0, NULL, NULL)
			       != CL_SUCCESS)
			return fail("push host-write buffer: %d", -1);
	}
	for (int i = 0; i < n; i++) {
		const struct cmd *c = &list[i];
		if (c->kind == K_GEMM) {
			if (run_gemm(c))
				return -1;
		} else if (c->kind == K_UNARY || c->kind == K_UNARY2) {
			if (run_unary(c))
				return -1;
		} else if (c->kind == K_ROW) {
			if (run_row(c))
				return -1;
		} else if (c->kind == K_COPY) {
			struct buffer *s = slot(c->a), *t = slot(c->c);
			if (s->host && t->host) {
				memmove((char *)t->host + c->to, (char *)s->host + c->so, c->bytes);
			} else {
				cl_event ev = NULL;
				if (clEnqueueCopyBuffer(g.q, s->mem, t->mem, (size_t)c->so,
							(size_t)c->to, (size_t)c->bytes, 0, NULL,
							g.time ? &ev : NULL) != CL_SUCCESS)
					return fail("clEnqueueCopyBuffer failed");
				trace_push(ev, 4);
			}
		}
	}
	clFinish(g.q);
	trace_sum();
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
	free(g.graph[id].list);
	g.graph[id].live = 0;
	g.graph[id].list = NULL;
	g.graph[id].n = 0;
	return 0;
}

/* ---- plain GEMM path (gemm_runner) ------------------------------------------------ */
int xmx_init(const char *spv) { (void)spv; return g.open ? 0 : -1; }
int xmx_reserve(unsigned M, unsigned N, unsigned K, void **pa, void **pb, void **pc)
{
	(void)M; (void)N; (void)K; (void)pa; (void)pb; (void)pc;
	return fail("xmx_reserve is not used by the OpenCL backend");
}
int xmx_gemm(unsigned M, unsigned N, unsigned K, const void *a, const void *b, void *c,
	     unsigned iters)
{
	(void)M; (void)N; (void)K; (void)a; (void)b; (void)c; (void)iters;
	return fail("xmx_gemm (bare) not implemented; use the resident path");
}
int xmx_reserve_bytes(unsigned long long a, unsigned long long b, unsigned long long c,
		      void **pa, void **pb, void **pc)
{
	(void)a; (void)b; (void)c; (void)pa; (void)pb; (void)pc;
	return fail("xmx_reserve_bytes is not used by the OpenCL backend");
}
int xmx_init_batched(const char *spv) { (void)spv; return g.open ? 0 : -1; }
int xmx_gemm_batched(unsigned M, unsigned N, unsigned K, unsigned batch,
		     unsigned sa, unsigned sb, unsigned sc, unsigned bt)
{
	(void)M; (void)N; (void)K; (void)batch; (void)sa; (void)sb; (void)sc; (void)bt;
	return fail("xmx_gemm_batched is not used by the OpenCL backend");
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
	struct cmd *cm = &g.cmd[g.ncmd++];
	cm->kind = K_GEMM;
	cm->a = a; cm->b = b; cm->c = c; cm->d = skip; cm->second = cosine;
	cm->M = M; cm->N = N; cm->K = K; cm->batch = 1;
	cm->sa = M * K; cm->sb = K * N; cm->sc = M * N;
	cm->bt = flags | 0x20000u;
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
	struct cmd *cm = &g.cmd[g.ncmd++];
	cm->kind = K_GEMM;
	cm->a = a; cm->b = b; cm->c = c; cm->d = skip; cm->second = cosine;
	cm->M = M; cm->N = N; cm->K = K; cm->batch = 1;
	cm->sa = M * K; cm->sb = K * N; cm->sc = M * N;
	cm->height = h; cm->width = w; cm->across = ac; cm->pad = pad;
	cm->bt = flags | 0x20000u | 0x80000u;
	return 0;
}
int xmx_rec_gemm_qkv(int a, int weight, int q, int k, int v, int scale, unsigned M, unsigned channels, unsigned heads, unsigned tokens, unsigned vit)
{ (void)a;(void)weight;(void)q;(void)k;(void)v;(void)scale;(void)M;(void)channels;(void)heads;(void)tokens;(void)vit; return fail("gemm_qkv: not implemented yet"); }
int xmx_rec_gemm_qkv_window(int image, int weight, int q, int k, int v, int scale, unsigned M, unsigned channels, unsigned heads, unsigned tokens, unsigned width, unsigned height, unsigned across, unsigned pad, unsigned image_mode)
{ (void)image;(void)weight;(void)q;(void)k;(void)v;(void)scale;(void)M;(void)channels;(void)heads;(void)tokens;(void)width;(void)height;(void)across;(void)pad;(void)image_mode; return fail("gemm_qkv_window: not implemented yet"); }
int xmx_rec_gemm_dual(int a, int b, int c, int half_copy, unsigned M, unsigned N, unsigned K)
{ (void)a;(void)b;(void)c;(void)half_copy;(void)M;(void)N;(void)K; return fail("gemm_dual: not implemented yet"); }
static int rec_unary(unsigned kind, int a, int b, int c, int d, int second,
		     unsigned n, unsigned channels, float p0, unsigned batch,
		     unsigned sa, unsigned sb, unsigned sc, unsigned k)
{
	if (!slot(a) || !slot(c))
		return fail("unary: bad buffer");
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = &g.cmd[g.ncmd++];
	cm->kind = second >= 0 ? K_UNARY2 : K_UNARY;
	cm->flags = kind;
	cm->a = a; cm->b = b; cm->c = c; cm->d = d; cm->second = second;
	cm->n = n; cm->channels = channels; cm->cap = p0;
	cm->batch = batch; cm->sa = sa; cm->sb = sb; cm->sc = sc; cm->k = k;
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
	if (clEnqueueNDRangeKernel(g.q, k, 1, NULL, &global, &local, 0, NULL,
				   g.time ? &ev : NULL) != CL_SUCCESS)
		return fail("unary dispatch");
	trace_push(ev, c->kind == 2 ? 1 : 2);
	return 0;
}
int xmx_rec_row(unsigned kind, int a, int b, int c, int d, unsigned rows, unsigned width, unsigned heads, unsigned scaled, unsigned stride, float cap)
{
	if (!slot(a) || !slot(c))
		return fail("row: bad buffer");
	if (g.ncmd >= MAX_CMD)
		return fail("command list full");
	struct cmd *cm = &g.cmd[g.ncmd++];
	cm->kind = K_ROW;
	cm->flags = kind;
	cm->a = a; cm->b = b; cm->c = c; cm->d = d;
	cm->M = rows; cm->N = width; cm->batch = heads; cm->k = scaled; cm->sa = stride;
	cm->cap = cap;
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
	if (clEnqueueNDRangeKernel(g.q, k, 1, NULL, &global, &local, 0, NULL,
				   g.time ? &ev : NULL) != CL_SUCCESS)
		return fail("row dispatch");
	trace_push(ev, 3);
	return 0;
}
int xmx_rec_history(int history, int motion, int out, unsigned pixels, unsigned channels, unsigned height, unsigned width, unsigned absolute)
{ (void)history;(void)motion;(void)out;(void)pixels;(void)channels;(void)height;(void)width;(void)absolute; return fail("history: not implemented yet"); }
int xmx_rec_qkv(int source, int q, int k, int v, int scale, unsigned rows, unsigned tokens, unsigned heads, unsigned vit)
{ (void)source;(void)q;(void)k;(void)v;(void)scale;(void)rows;(void)tokens;(void)heads;(void)vit; return fail("qkv: not implemented yet"); }
int xmx_rec_window_attention(int q, int k, int v, int bias, int out, unsigned batches, unsigned heads, unsigned merged)
{ (void)q;(void)k;(void)v;(void)bias;(void)out;(void)batches;(void)heads;(void)merged; return fail("window_attention: not implemented yet"); }
int xmx_rec_ffn(int a, int expand, int projection, int out, int skip, int cosine, unsigned M, unsigned cin, unsigned hidden, unsigned groups, unsigned flags)
{ (void)a;(void)expand;(void)projection;(void)out;(void)skip;(void)cosine;(void)M;(void)cin;(void)hidden;(void)groups;(void)flags; return fail("ffn: not implemented yet"); }
int xmx_rec_ffn_merge(int source, int skip, int sincos, int expand, int projection, int out, int cosine, unsigned height, unsigned width, unsigned source_width, unsigned flags)
{ (void)source;(void)skip;(void)sincos;(void)expand;(void)projection;(void)out;(void)cosine;(void)height;(void)width;(void)source_width;(void)flags; return fail("ffn_merge: not implemented yet"); }
int xmx_rec_ffn_stem(int features, int adapter, int expand, int projection, int out, int cosine, unsigned M, unsigned flags)
{ (void)features;(void)adapter;(void)expand;(void)projection;(void)out;(void)cosine;(void)M;(void)flags; return fail("ffn_stem: not implemented yet"); }
int xmx_rec_gemm_window_residual_pool(int a, int b, int skip_out, int skip, int cosine, int pooled, unsigned M, unsigned N, unsigned K, unsigned flags, unsigned height, unsigned width, unsigned across, unsigned pad)
{ (void)a;(void)b;(void)skip_out;(void)skip;(void)cosine;(void)pooled;(void)M;(void)N;(void)K;(void)flags;(void)height;(void)width;(void)across;(void)pad; return fail("gemm_window_residual_pool: not implemented yet"); }
int xmx_rec_window_block(int image, int qkv, int projection, int target, int bias, int cosine, int scale, int pooled, unsigned windows, unsigned height, unsigned width, unsigned across, unsigned pad, unsigned flags)
{ (void)image;(void)qkv;(void)projection;(void)target;(void)bias;(void)cosine;(void)scale;(void)pooled;(void)windows;(void)height;(void)width;(void)across;(void)pad;(void)flags; return fail("window_block: not implemented yet"); }
int xmx_rec_global_attention(int q, int k, int v, int merged, unsigned rows, unsigned tokens, unsigned heads)
{ (void)q;(void)k;(void)v;(void)merged;(void)rows;(void)tokens;(void)heads; return fail("global_attention: not implemented yet"); }
int xmx_rec_gemm_int8(int a, int b, int c, int a_scale, int b_scale, unsigned M, unsigned N, unsigned K, unsigned flags)
{ (void)a;(void)b;(void)c;(void)a_scale;(void)b_scale;(void)M;(void)N;(void)K;(void)flags; return fail("gemm_int8: not implemented yet"); }

/* ---- profiling -------------------------------------------------------------------- */
int xmx_profile(int on) { (void)on; return 0; }
void xmx_profile_reset(void) {}
double xmx_profile_ms(unsigned kind) { (void)kind; return 0.0; }
unsigned xmx_profile_count(unsigned kind) { (void)kind; return 0; }
unsigned xmx_profile_each_count(void) { return 0; }
double xmx_profile_each_ms(unsigned i) { (void)i; return 0.0; }
unsigned xmx_profile_each_kind(unsigned i) { (void)i; return 0; }