#!/usr/bin/env python3
"""xmx_throughput - sustained GEMM throughput of the cooperative-matrix path.

`bench.py` times a fixed number of dispatches, which on a clock-gated GPU measures the
ramp as much as the kernel: a shape that finishes in a few milliseconds reports a
fraction of its steady rate. This one runs each shape for a wall-clock budget
(`NR_BENCH_SECONDS`, default 8) after a warm-up, and reports the rate over the whole
sustained interval. It uses the same entry point the graph uses (`xmx_gemm`), so it
measures the kernel libxmx actually dispatches - including the device's own tile N
(16 on Xe2, 8 on Xe-HPG; `xmx_coopmat_n`).

Every shape is also checked against a float32 reference, so a kernel that runs to the
wrong answer cannot be reported as fast.
"""

import os
import sys
import time

import numpy as np
import xmx

BUDGET = float(os.environ.get("NR_BENCH_SECONDS", "8"))
TILE = getattr(xmx, "TN", 16)

lib = xmx._load()
print(f"device: {xmx.device_name()}")
print(f"path:   {xmx.path_note()}")
print("tile:   %dx%d" % (getattr(xmx, "TM", 8), xmx._TN))
print(f"budget: {BUDGET:.1f} s/shape sustained\n")


def one(M, N, K, seconds):
    """Run MxNxK GEMMs continuously for `seconds`; return (ms/dispatch, GFLOP/s, C)."""
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray((rng.standard_normal((M, K)) * 0.1).astype(np.float16))
    B = np.ascontiguousarray((rng.standard_normal((K, N)) * 0.1).astype(np.float16))
    C = np.empty((M, N), dtype=np.float32)
    flops = 2.0 * M * N * K
    # One call's worth of dispatches, sized so a submit lasts long enough that host
    # round trips are noise: at least 30 ms of work, then repeat until the budget.
    lib.xmx_gemm(M, N, K, A.ctypes.data, B.ctypes.data, C.ctypes.data, 1)  # warm
    t = time.perf_counter()
    lib.xmx_gemm(M, N, K, A.ctypes.data, B.ctypes.data, C.ctypes.data, 8)
    single = (time.perf_counter() - t) / 8.0
    iters = max(1, int(0.03 / max(single, 1e-6)))
    lib.xmx_gemm(M, N, K, A.ctypes.data, B.ctypes.data, C.ctypes.data, iters)  # steady warm
    calls = 0
    t0 = time.perf_counter()
    while True:
        if lib.xmx_gemm(M, N, K, A.ctypes.data, B.ctypes.data, C.ctypes.data, iters) != 0:
            raise SystemExit(f"xmx_gemm failed: {lib.xmx_error().decode()}")
        calls += iters
        if time.perf_counter() - t0 >= seconds:
            break
    dt = time.perf_counter() - t0
    ref = A.astype(np.float32) @ B.astype(np.float32)
    err = np.abs(C.astype(np.float64) - ref.astype(np.float64)).max() / max(
        np.abs(ref).max(), 1e-30
    )
    return 1e3 * dt / calls, flops * calls / dt / 1e9, err


SHAPES = [
    (512, 512, 512),
    (1024, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 512),
    (4096, 512, 512),
    (8192, 512, 512),
    (16384, 512, 512),
    (4096, 1024, 4096),
]
if len(sys.argv) > 1:
    SHAPES = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]]

print("  %-20s %12s %12s %10s" % ("shape", "ms/dispatch", "GFLOP/s", "rel err"))
worst = 0.0
peak = 0.0
for M, N, K in SHAPES:
    ms, gf, err = one(M, N, K, BUDGET)
    worst = max(worst, err)
    peak = max(peak, gf)
    print("  %-20s %12.3f %12.1f %10.2e" % ("%dx%dx%d" % (M, N, K), ms, gf, err))
print(f"\n  peak {peak:.1f} GFLOP/s sustained; worst relative error {worst:.2e}")
raise SystemExit(0 if worst < 1e-5 else 1)
