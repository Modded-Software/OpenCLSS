#!/usr/bin/env python3
"""DG2 (Xe-HPG) DPAS through OpenCL: subgroup 8, the 8x8x16 atom, register-blocked.

The Vulkan cooperative-matrix path forces subgroupSize=32 on a chip whose DPAS exec size
is 8, which is the register ceiling phase26 describes (an 8x16 fp32 accumulator is 4
registers a lane at SIMD32, so any real block spills).  OpenCL lets the kernel demand
subgroup 8, where a DPAS accumulator is one register a lane and a 32x64 or 64x64 block
fits.  Intel's own tuning note names 32x64x16 as the optimum.

Driven through ctypes (no CL headers on the box).  Every entry point declared: without
argtypes ctypes truncates handles and the first call takes the process with it.
"""

import argparse
import ctypes
import os
import pathlib
import sys
import time

import numpy as np

VECFLAG = "-DVEC" if os.environ.get("VEC") else ""
# CL_SRC picks the kernel file beside this one; a "blk" kernel takes B VNNI-packed.
CL_SRC = os.environ.get("CL_SRC", "opencl_gemm8.cl")
PACKED = "blk" in CL_SRC


def pack_vnni(B):
    """(batch, K, N) half -> the blk kernel's fragments: [batch][K/16][N/8][i][lane] uint32
    holding (B[k0 + 2i][n], B[k0 + 2i + 1][n]), the even k in the low half."""
    batch, K, N = B.shape
    t = B.reshape(batch, K // 16, 8, 2, N // 8, 8).transpose(0, 1, 4, 2, 5, 3)
    return np.ascontiguousarray(t).view(np.uint32).reshape(batch, K * N // 2)


# Debian ships only the soname (libOpenCL.so.1) unless the -dev package is installed.
try:
    CL = ctypes.CDLL("libOpenCL.so")
except OSError:
    CL = ctypes.CDLL("libOpenCL.so.1")
V, U, I, S, L = (
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_ulonglong,
)
PI, PU = ctypes.POINTER(I), ctypes.POINTER(U)
CL.clGetPlatformIDs.argtypes, CL.clGetPlatformIDs.restype = [U, V, PU], I
CL.clGetDeviceIDs.argtypes, CL.clGetDeviceIDs.restype = [V, L, U, V, PU], I
CL.clCreateContext.argtypes, CL.clCreateContext.restype = [V, U, V, V, V, PI], V
CL.clCreateCommandQueue.argtypes, CL.clCreateCommandQueue.restype = [V, V, L, PI], V
CL.clCreateProgramWithSource.argtypes, CL.clCreateProgramWithSource.restype = (
    [V, U, V, V, PI],
    V,
)
CL.clBuildProgram.argtypes, CL.clBuildProgram.restype = (
    [V, U, V, ctypes.c_char_p, V, V],
    I,
)
CL.clGetProgramBuildInfo.argtypes, CL.clGetProgramBuildInfo.restype = (
    [V, V, U, S, V, V],
    I,
)
CL.clGetDeviceInfo.argtypes, CL.clGetDeviceInfo.restype = [V, U, S, V, V], I
CL.clCreateKernel.argtypes, CL.clCreateKernel.restype = [V, ctypes.c_char_p, PI], V
CL.clGetKernelWorkGroupInfo.argtypes, CL.clGetKernelWorkGroupInfo.restype = (
    [V, V, U, S, V, V],
    I,
)
CL.clCreateBuffer.argtypes, CL.clCreateBuffer.restype = [V, L, S, V, PI], V
CL.clSetKernelArg.argtypes, CL.clSetKernelArg.restype = [V, U, S, V], I
CL.clEnqueueNDRangeKernel.argtypes, CL.clEnqueueNDRangeKernel.restype = (
    [V, V, U, V, V, V, U, V, V],
    I,
)
CL.clEnqueueReadBuffer.argtypes, CL.clEnqueueReadBuffer.restype = (
    [V, V, U, S, S, V, U, V, V],
    I,
)
CL.clFinish.argtypes, CL.clFinish.restype = [V], I

GPU, READ_ONLY, WRITE_ONLY, COPY_HOST = 1 << 2, 1 << 2, 1 << 1, 1 << 5
BUILD_LOG = 0x1183
DEVICE_NAME = 0x102B
PRIVATE_MEM = 0x11B4
REGISTER_COUNT = 0x11B9  # CL_KERNEL_SPILL_MEM_SIZE (Intel), if the driver answers it


def kernel_private_bytes(device, kernel):
    value = L()
    if CL.clGetKernelWorkGroupInfo(
        kernel, device, PRIVATE_MEM, ctypes.sizeof(value), ctypes.byref(value), None
    ):
        return -1
    return value.value


def kernel_spill_bytes(device, kernel):
    value = L()
    if CL.clGetKernelWorkGroupInfo(
        kernel, device, REGISTER_COUNT, ctypes.sizeof(value), ctypes.byref(value), None
    ):
        return -1
    return value.value


def device_and_context(index):
    platforms, count = (V * 4)(), U()
    CL.clGetPlatformIDs(4, platforms, ctypes.byref(count))
    for p in range(count.value):
        devices, found = (V * 4)(), U()
        if (
            not CL.clGetDeviceIDs(V(platforms[p]), GPU, 4, devices, ctypes.byref(found))
            and found.value
        ):
            if index >= found.value:
                continue
            device, err = V(devices[index]), I()
            buf = ctypes.create_string_buffer(256)
            CL.clGetDeviceInfo(device, DEVICE_NAME, 256, buf, None)
            print(f"device: {buf.value.decode()}")
            context = CL.clCreateContext(
                None, 1, ctypes.byref(device), None, None, ctypes.byref(err)
            )
            if err.value:
                raise RuntimeError(f"clCreateContext: {err.value}")
            return device, context
    raise RuntimeError("no OpenCL GPU")


def compile_kernel(context, device, source, options):
    err = I()
    text = source.encode()
    array, length = ctypes.c_char_p(text), S(len(text))
    program = CL.clCreateProgramWithSource(
        context, 1, ctypes.byref(array), ctypes.byref(length), ctypes.byref(err)
    )
    if err.value:
        raise RuntimeError(f"clCreateProgramWithSource: {err.value}")
    if CL.clBuildProgram(program, 1, ctypes.byref(device), options.encode(), None, None):
        size = S()
        CL.clGetProgramBuildInfo(program, device, BUILD_LOG, 0, None, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        CL.clGetProgramBuildInfo(program, device, BUILD_LOG, size.value, log, None)
        message = log.value.decode(errors="replace").strip()
        if not message:
            message = "clBuildProgram failed with no log"
        raise RuntimeError(message)
    kernel = CL.clCreateKernel(program, b"gemm", ctypes.byref(err))
    if err.value:
        raise RuntimeError(f"clCreateKernel: {err.value}")
    return kernel


def run(device, context, M, N, K, RM, RN, KU, source, repeats=5, calls=40, batch=1):
    kernel = compile_kernel(
        context, device, source, f"-DRM={RM} -DRN={RN} -DKUNROLL={KU} {VECFLAG}"
    )
    err = I()
    queue = CL.clCreateCommandQueue(context, device, 0, ctypes.byref(err))
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray((rng.standard_normal((batch, M, K)) * 0.1).astype(np.float16))
    B = np.ascontiguousarray((rng.standard_normal((batch, K, N)) * 0.1).astype(np.float16))
    C = np.zeros((batch, M, N), np.float32)

    def buffer(array, flags):
        handle = CL.clCreateBuffer(
            context,
            flags | COPY_HOST,
            array.nbytes,
            array.ctypes.data,
            ctypes.byref(err),
        )
        if err.value:
            raise RuntimeError(f"clCreateBuffer: {err.value}")
        return V(handle)

    Bdev = pack_vnni(B) if PACKED else B
    da, db, dc = buffer(A, READ_ONLY), buffer(Bdev, READ_ONLY), buffer(C, WRITE_ONLY)
    for index, handle in ((0, da), (1, db), (2, dc)):
        CL.clSetKernelArg(kernel, index, ctypes.sizeof(V), ctypes.byref(handle))
    for index, value in ((3, M), (4, N), (5, K)):
        scalar = U(value)
        CL.clSetKernelArg(kernel, index, ctypes.sizeof(scalar), ctypes.byref(scalar))
    work = (S * 3)(N // (8 * RN) * 8, M // (8 * RM), batch)
    group = (S * 3)(8, 1, 1)

    def once():
        for _ in range(calls):
            status = CL.clEnqueueNDRangeKernel(queue, kernel, 2, None, work, group, 0, None, None)
            if status:
                raise RuntimeError(f"clEnqueueNDRangeKernel: {status}")
        CL.clFinish(queue)

    once()
    best = min(
        (lambda t=time.perf_counter(): (once(), time.perf_counter() - t)[1])()
        for _ in range(repeats)
    )
    return (
        2.0 * M * N * K * calls / best / 1e9,
        best,
        kernel_private_bytes(device, kernel),
        kernel_spill_bytes(device, kernel),
    )


def readback(device, context, M, N, K, RM, RN, KU, source, batch=1):
    # Re-run one plain call and copy C back to judge correctness against numpy.
    kernel = compile_kernel(
        context, device, source, f"-DRM={RM} -DRN={RN} -DKUNROLL={KU} {VECFLAG}"
    )
    err = I()
    queue = CL.clCreateCommandQueue(context, device, 0, ctypes.byref(err))
    rng = np.random.default_rng(3)
    A = np.ascontiguousarray((rng.standard_normal((batch, M, K)) * 0.1).astype(np.float16))
    B = np.ascontiguousarray((rng.standard_normal((batch, K, N)) * 0.1).astype(np.float16))
    out = np.zeros((batch, M, N), np.float32)

    def buffer(array, flags):
        h = CL.clCreateBuffer(
            context,
            flags | COPY_HOST,
            array.nbytes,
            array.ctypes.data,
            ctypes.byref(err),
        )
        return V(h)

    Bdev = pack_vnni(B) if PACKED else B
    da, db, dc = buffer(A, READ_ONLY), buffer(Bdev, READ_ONLY), buffer(out, WRITE_ONLY)
    for i, h in ((0, da), (1, db), (2, dc)):
        CL.clSetKernelArg(kernel, i, ctypes.sizeof(V), ctypes.byref(h))
    for i, v in ((3, M), (4, N), (5, K)):
        s = U(v)
        CL.clSetKernelArg(kernel, i, ctypes.sizeof(s), ctypes.byref(s))
    work = (S * 3)(N // (8 * RN) * 8, M // (8 * RM), batch)
    group = (S * 3)(8, 1, 1)
    CL.clEnqueueNDRangeKernel(queue, kernel, 3, None, work, group, 0, None, None)
    CL.clFinish(queue)
    CL.clEnqueueReadBuffer(queue, dc, 1, 0, out.nbytes, out.ctypes.data, 0, None, None)
    CL.clFinish(queue)
    want = np.einsum("bmk,bkn->bmn", A.astype(np.float32), B.astype(np.float32))
    scale = max(float(np.abs(want).max()), 1e-9)
    per_batch = [float(np.abs(out[b] - want[b]).max() / scale) for b in range(batch)]
    return max(per_batch), per_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="1x1,2x2,4x4,8x8")
    ap.add_argument("--shape", default="192,4096,1024")
    ap.add_argument("--kunroll", type=int, default=1)
    ap.add_argument("--device", type=int, default=int(os.environ.get("XMX_DEVICE", "0")))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    M, N, K = (int(v) for v in args.shape.split(","))
    source = pathlib.Path(__file__).with_name(CL_SRC).read_text()
    device, context = device_and_context(args.device)
    print(f"  {M}x{N}x{K}, subgroup 8, DG2 8x8x16 DPAS, register-blocked (RM x RN atoms)")
    print(
        "  Vulkan cooperative matrix on this card peaks at ~10800 GFLOP/s; XMX FP16 peak ~137000\n"
    )
    print(
        "  %-10s %8s %12s %12s %10s %10s %10s"
        % ("block", "atoms", "GFLOP/s", "ms", "rel err", "priv", "spill")
    )
    failures = []
    for block in args.blocks.split(","):
        rm, rn = (int(v) for v in block.split("x"))
        if M % (8 * rm) or N % (8 * rn) or K % (16 * args.kunroll):
            continue
        try:
            rate, best, priv, spill = run(device, context, M, N, K, rm, rn, args.kunroll, source)
            err = (
                readback(device, context, M, N, K, rm, rn, args.kunroll, source)[0]
                if args.check
                else float("nan")
            )
            print(
                "  %-10s %7d %12.1f %12.4f %10.2e %10d %10d"
                % (
                    f"{8 * rm}x{8 * rn}",
                    rm * rn,
                    rate,
                    1000 * best / 40,
                    err,
                    priv,
                    spill,
                )
            )
        except RuntimeError as failure:
            failures.append((block, str(failure).splitlines()[0][:100]))
    if failures:
        for block, message in failures:
            print(f"  {block} FAILED: {message}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} block(s) failed to run")


if __name__ == "__main__":
    main()
