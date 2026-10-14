#!/usr/bin/env python3
"""
test_gemm - check the XMX cooperative-matrix GEMM against numpy.

The GPU path must reproduce FP16 operands with FP32 accumulation. The reference is
computed in float64 so the comparison measures the GPU, not the reference.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xmx

import nr_build

RUNNER = nr_build.executable("gemm_runner")

# The shader follows the device: the cooperative-matrix kernel where the extension
# exists, the portable multiply-add one where it does not (or XMX_PORTABLE=1). The
# cooperative tile's N is the device's, so Xe-HPG (DG2) takes the 8x8 build and Xe2
# the 8x16 one; `xmx` reads the device's N at open (`_TN`).
if xmx.portable():
    SPV = nr_build.shader("gemm_portable_desc.spv")
else:
    SPV = nr_build.shader("gemm_coopmat_n8.spv" if xmx._TN == 8 else "gemm_coopmat.spv")


def run(M, N, K, A, B):
    # a context manager, not mkdtemp: this is called once per shape and the operand
    # dumps are megabytes, on a machine whose /tmp is RAM it cannot spare
    with tempfile.TemporaryDirectory(prefix="nr-gemm-") as temporary:
        d = Path(temporary)
        (d / "a").write_bytes(A.astype("<f2").tobytes())
        (d / "b").write_bytes(B.astype("<f2").tobytes())
        r = subprocess.run(
            [
                str(RUNNER),
                str(SPV),
                str(M),
                str(N),
                str(K),
                str(d / "a"),
                str(d / "b"),
                str(d / "c"),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit("gemm_runner failed (%d)" % r.returncode)
        return (
            np.frombuffer((d / "c").read_bytes(), dtype="<f4").reshape(M, N),
            r.stderr.strip(),
        )


def check(M, N, K, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, K)) * scale).astype(np.float16)
    B = (rng.standard_normal((K, N)) * scale).astype(np.float16)
    gpu, _log = run(M, N, K, A, B)
    ref32 = A.astype(np.float32) @ B.astype(np.float32)  # the intended contract
    ref64 = A.astype(np.float64) @ B.astype(np.float64)  # exact
    dev = np.abs(gpu.astype(np.float64) - ref64).max()
    rel = dev / max(np.abs(ref64).max(), 1e-30)
    same = np.array_equal(gpu, ref32)
    print(
        "  %4dx%-4d K=%-5d  max|gpu-exact|=%-11.4g rel=%-11.4g  bitwise==fp32 numpy: %s"
        % (M, N, K, dev, rel, same)
    )
    return rel


def main():
    print(f"=== XMX GEMM vs numpy - {xmx.path_note()} ===")
    print("device line from the runner:")
    _, log = run(8, 16, 16, np.zeros((8, 16), np.float16), np.zeros((16, 16), np.float16))
    for l in log.splitlines():
        print("   " + l)
    print()
    worst = 0.0
    for M, N, K in [
        (8, 16, 16),
        (8, 16, 64),
        (16, 32, 64),
        (64, 64, 256),
        (128, 256, 512),
        (256, 512, 512),
    ]:
        worst = max(worst, check(M, N, K))
    print(f"\n  worst relative error across all shapes: {worst:.4g}")
    return 0 if worst < 1e-5 else 1


raise SystemExit(main())
