#!/usr/bin/env python3
"""SLM-staged DG2 subgroup-8 DPAS GEMM (src/bench/opencl_gemm8_slm.cl).

Workgroup tile BM x BN x BK, SGM x SGN subgroups of 8, each subgroup an RM x RN block of
8x8x16 atoms.  Same ctypes harness as opencl_gemm8.py.
"""
import argparse, ctypes, pathlib, time
import numpy as np

CL = ctypes.CDLL("libOpenCL.so")
V, U, I, S, L = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_size_t, ctypes.c_ulonglong
PI, PU = ctypes.POINTER(I), ctypes.POINTER(U)
CL.clGetPlatformIDs.argtypes, CL.clGetPlatformIDs.restype = [U, V, PU], I
CL.clGetDeviceIDs.argtypes, CL.clGetDeviceIDs.restype = [V, L, U, V, PU], I
CL.clCreateContext.argtypes, CL.clCreateContext.restype = [V, U, V, V, V, PI], V
CL.clCreateCommandQueue.argtypes, CL.clCreateCommandQueue.restype = [V, V, L, PI], V
CL.clCreateProgramWithSource.argtypes, CL.clCreateProgramWithSource.restype = [V, U, V, V, PI], V
CL.clBuildProgram.argtypes, CL.clBuildProgram.restype = [V, U, V, ctypes.c_char_p, V, V], I
CL.clGetProgramBuildInfo.argtypes, CL.clGetProgramBuildInfo.restype = [V, V, U, S, V, V], I
CL.clGetDeviceInfo.argtypes, CL.clGetDeviceInfo.restype = [V, U, S, V, V], I
CL.clCreateKernel.argtypes, CL.clCreateKernel.restype = [V, ctypes.c_char_p, PI], V
CL.clCreateBuffer.argtypes, CL.clCreateBuffer.restype = [V, L, S, V, PI], V
CL.clSetKernelArg.argtypes, CL.clSetKernelArg.restype = [V, U, S, V], I
CL.clEnqueueNDRangeKernel.argtypes, CL.clEnqueueNDRangeKernel.restype = [V, V, U, V, V, V, U, V, V], I
CL.clEnqueueReadBuffer.argtypes, CL.clEnqueueReadBuffer.restype = [V, V, U, S, S, V, U, V, V], I
CL.clFinish.argtypes, CL.clFinish.restype = [V], I

GPU, READ_ONLY, WRITE_ONLY, COPY_HOST = 1 << 2, 1 << 2, 1 << 1, 1 << 5
BUILD_LOG, DEVICE_NAME = 0x1183, 0x102B


def device_and_context(index):
    platforms, count = (V * 4)(), U()
    CL.clGetPlatformIDs(4, platforms, ctypes.byref(count))
    for p in range(count.value):
        devices, found = (V * 4)(), U()
        if not CL.clGetDeviceIDs(V(platforms[p]), GPU, 4, devices, ctypes.byref(found)) and found.value:
            if index >= found.value:
                continue
            device, err = V(devices[index]), I()
            buf = ctypes.create_string_buffer(256)
            CL.clGetDeviceInfo(device, DEVICE_NAME, 256, buf, None)
            print("device: %s" % buf.value.decode())
            context = CL.clCreateContext(None, 1, ctypes.byref(device), None, None, ctypes.byref(err))
            if err.value:
                raise RuntimeError(f"clCreateContext: {err.value}")
            return device, context
    raise RuntimeError("no OpenCL GPU")


def compile_kernel(context, device, source, options):
    err = I()
    text = source.encode()
    array, length = ctypes.c_char_p(text), S(len(text))
    program = CL.clCreateProgramWithSource(context, 1, ctypes.byref(array), ctypes.byref(length), ctypes.byref(err))
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


def measure(device, context, M, N, K, BM, BN, BK, RM, RN, source, repeats=5, calls=30, check=False):
    opts = f"-DBM={BM} -DBN={BN} -DBK={BK} -DRM={RM} -DRN={RN}"
    kernel = compile_kernel(context, device, source, opts)
    err = I()
    queue = CL.clCreateCommandQueue(context, device, 0, ctypes.byref(err))
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray((rng.standard_normal((M, K)) * 0.1).astype(np.float16))
    B = np.ascontiguousarray((rng.standard_normal((K, N)) * 0.1).astype(np.float16))
    C = np.zeros((M, N), np.float32)

    def buffer(array, flags):
        return V(CL.clCreateBuffer(context, flags | COPY_HOST, array.nbytes, array.ctypes.data, ctypes.byref(err)))

    da, db, dc = buffer(A, READ_ONLY), buffer(B, READ_ONLY), buffer(C, WRITE_ONLY)
    for i, h in ((0, da), (1, db), (2, dc)):
        CL.clSetKernelArg(kernel, i, ctypes.sizeof(V), ctypes.byref(h))
    for i, val in ((3, M), (4, N), (5, K)):
        s = U(val)
        CL.clSetKernelArg(kernel, i, ctypes.sizeof(s), ctypes.byref(s))
    wgs = 8 * (BM // (8 * RM)) * (BN // (8 * RN))
    work = (S * 2)(N // BN * wgs, M // BM)
    group = (S * 2)(wgs, 1)

    def once():
        for _ in range(calls):
            status = CL.clEnqueueNDRangeKernel(queue, kernel, 2, None, work, group, 0, None, None)
            if status:
                raise RuntimeError(f"clEnqueueNDRangeKernel: {status}")
        CL.clFinish(queue)

    once()
    best = min((lambda t=time.perf_counter(): (once(), time.perf_counter() - t)[1])() for _ in range(repeats))
    errp = float("nan")
    if check:
        CL.clEnqueueReadBuffer(queue, dc, 1, 0, C.nbytes, C.ctypes.data, 0, None, None)
        CL.clFinish(queue)
        want = A.astype(np.float32) @ B.astype(np.float32)
        errp = float(np.abs(C - want).max() / max(float(np.abs(want).max()), 1e-9))
    return 2.0 * M * N * K * calls / best / 1e9, 1000 * best / calls, errp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="192,4096,1024")
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--bk", type=int, default=32)
    ap.add_argument("--rm", type=int, default=2)
    ap.add_argument("--rn", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    M, N, K = (int(v) for v in args.shape.split(","))
    source = pathlib.Path(__file__).with_name("opencl_gemm8_slm.cl").read_text()
    device, context = device_and_context(args.device)
    print(f"  {M}x{N}x{K}, subgroup 8, SLM-staged DPAS")
    if M % args.bm or N % args.bn or K % args.bk:
        raise SystemExit(f"shape not divisible by BM={args.bm} BN={args.bn} BK={args.bk}")
    if args.bm % (8 * args.rm) or args.bn % (8 * args.rn):
        raise SystemExit("BM/BN not divisible by 8*RM / 8*RN")
    rate, ms, errp = measure(device, context, M, N, K, args.bm, args.bn, args.bk,
                             args.rm, args.rn, source, check=args.check)
    wgs = 8 * (args.bm // (8 * args.rm)) * (args.bn // (8 * args.rn))
    print("  WG %dx%dx%d  SGM=%d SGN=%d  WGS=%d  RMxRN=%dx%d : %.1f GFLOP/s  %.4f ms  rel err %.2e"
          % (args.bm, args.bn, args.bk, args.bm // (8 * args.rm), args.bn // (8 * args.rn),
             wgs, args.rm, args.rn, rate, ms, errp))


if __name__ == "__main__":
    main()