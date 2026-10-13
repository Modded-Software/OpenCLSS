#!/usr/bin/env python3
"""Deterministic SLM-staged FMA GEMM (src/bench/opencl_gemm8_fma_slm.cl).

Same ctypes harness as opencl_gemm8_slm.py; the kernel under test is the staged FMA core
rather than the staged DPAS one. Compare against the shipping scalar FMA kernel with
gemm_sustained.py on the same shapes.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import opencl_gemm8_slm as H


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
    source = pathlib.Path(__file__).with_name("opencl_gemm8_fma_slm.cl").read_text()
    device, context = H.device_and_context(args.device)
    if M % args.bm or N % args.bn or K % args.bk:
        raise SystemExit(f"shape not divisible by BM={args.bm} BN={args.bn} BK={args.bk}")
    if args.bm % (8 * args.rm) or args.bn % (8 * args.rn):
        raise SystemExit("BM/BN not divisible by 8*RM / 8*RN")
    rate, ms, errp = H.measure(device, context, M, N, K, args.bm, args.bn, args.bk,
                               args.rm, args.rn, source, check=args.check)
    wgs = 8 * (args.bm // (8 * args.rm)) * (args.bn // (8 * args.rn))
    print("  %dx%dx%d  WG %dx%dx%d  SGM=%d SGN=%d WGS=%d RMxRN=%dx%d : %.1f GFLOP/s  %.4f ms  rel err %.2e"
          % (M, N, K, args.bm, args.bn, args.bk, args.bm // (8 * args.rm),
             args.bn // (8 * args.rn), wgs, args.rm, args.rn, rate, ms, errp))


if __name__ == "__main__":
    main()