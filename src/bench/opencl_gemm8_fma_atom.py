#!/usr/bin/env python3
"""Scalar FMA atom GEMM (src/bench/opencl_gemm8_fma_atom.cl) in the same harness as the
staged kernels, so the staged design can be compared against the shipping core directly."""
import argparse
import ctypes
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import opencl_gemm8_slm as H

V, U, I, S, L = H.V, H.U, H.I, H.S, H.L
CL = H.CL


def measure(device, context, M, N, K, RM, RN, source, repeats=5, calls=30, check=False):
    opts = f"-DRM={RM} -DRN={RN}"
    kernel = H.compile_kernel(context, device, source, opts)
    err = I()
    queue = CL.clCreateCommandQueue(context, device, 0, ctypes.byref(err))
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray((rng.standard_normal((M, K)) * 0.1).astype(np.float16))
    B = np.ascontiguousarray((rng.standard_normal((K, N)) * 0.1).astype(np.float16))
    C = np.zeros((M, N), np.float32)

    def buffer(array, flags):
        return V(CL.clCreateBuffer(context, flags | H.COPY_HOST, array.nbytes,
                                   array.ctypes.data, ctypes.byref(err)))

    da, db, dc = buffer(A, H.READ_ONLY), buffer(B, H.READ_ONLY), buffer(C, H.WRITE_ONLY)
    for i, h in ((0, da), (1, db), (2, dc)):
        CL.clSetKernelArg(kernel, i, ctypes.sizeof(V), ctypes.byref(h))
    for i, val in ((3, M), (4, N), (5, K)):
        s = U(val)
        CL.clSetKernelArg(kernel, i, ctypes.sizeof(s), ctypes.byref(s))
    work = (S * 2)(N // (8 * RN) * 8, M // (8 * RM))
    group = (S * 2)(8, 1)

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
    ap.add_argument("--rm", type=int, default=2)
    ap.add_argument("--rn", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    M, N, K = (int(v) for v in args.shape.split(","))
    source = pathlib.Path(__file__).with_name("opencl_gemm8_fma_atom.cl").read_text()
    device, context = H.device_and_context(args.device)
    rate, ms, errp = measure(device, context, M, N, K, args.rm, args.rn, source, check=args.check)
    print("  %dx%dx%d  atom RMxRN=%dx%d : %.1f GFLOP/s  %.4f ms  rel err %.2e"
          % (M, N, K, args.rm, args.rn, rate, ms, errp))


if __name__ == "__main__":
    main()