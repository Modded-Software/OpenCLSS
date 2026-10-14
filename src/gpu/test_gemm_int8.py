#!/usr/bin/env python3
"""The integer cooperative-matrix kernel against numpy, with no tolerance at all.

`gemm_coopmat_int8.comp` is configuration 4: `sint8 x sint8 -> sint32`. The accumulator
is a 32-bit integer, so the result is not an approximation of the dot product, it **is**
the dot product - no subnormal flush, no rounding between multiply and add, nothing that
`notes/phase9-numerics.md` says about float agreement applies here.

So this test demands exact equality. A single differing element is a bug in the kernel,
the layout or the padding, and not a precision story.
"""

import pathlib
import subprocess
import sys
import tempfile

import numpy as np

import nr_build

RUNNER = nr_build.executable("gemm_runner")
SPV = nr_build.shader("gemm_coopmat_int8.spv")


def integer_tile_n():
    """The device's integer cooperative-matrix tile N, or None when there is none: a device
    without VK_KHR_cooperative_matrix (MoltenVK, the phones) cannot build the kernel at all."""
    import ctypes

    lib = ctypes.CDLL(str(nr_build.library("xmx")))
    if lib.xmx_open() != 0 or lib.xmx_coopmat() != 1:
        return None
    return int(lib.xmx_coopmat_int8_n())


TM, TN, TK = 8, 16, 32  # the shape the kernel is written for (Xe2 config 4)
FAILURES = []


def check(name, ok, detail=""):
    print(
        f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
        flush=True,
    )
    if not ok:
        FAILURES.append(name)


def run(a, b):
    """One dispatch through the one-shot runner; returns C as int32."""
    m, k = a.shape
    n = b.shape[1]
    with tempfile.TemporaryDirectory() as room:
        room = pathlib.Path(room)
        (room / "a").write_bytes(a.astype(np.int8).tobytes())
        (room / "b").write_bytes(b.astype(np.int8).tobytes())
        done = subprocess.run(
            [
                str(RUNNER),
                str(SPV),
                str(m),
                str(n),
                str(k),
                str(room / "a"),
                str(room / "b"),
                str(room / "c"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if done.returncode != 0:
            raise RuntimeError((done.stderr.strip().splitlines() or ["no output"])[-1])
        return np.frombuffer((room / "c").read_bytes(), np.int32).reshape(m, n)


def main():
    tile_n = integer_tile_n()
    if nr_build.backend() != "vulkan" or tile_n is None:
        print(
            "gemm int8: skipped (the integer kernel needs VK_KHR_cooperative_matrix, which this "
            "device or runtime does not have) - a skip is not a pass"
        )
        return 0
    if tile_n != TN:
        print(
            f"gemm int8: skipped (this device's integer tile is N={tile_n}; the kernel is built "
            f"for N={TN}, Xe2's configuration 4. Xe-HPG exposes N=8 and needs an `_n8` twin, "
            "which does not exist yet) - a skip is not a pass"
        )
        return 0
    if not RUNNER.exists() or not SPV.exists():
        print(
            "gemm int8: skipped (make work/gemm_runner work/gemm_coopmat_int8.spv first)"
            " - a skip is not a pass"
        )
        return 0
    rng = np.random.default_rng(4)
    # The last two are the bottleneck's own shapes at render scale 0.35 and 0.55: small M,
    # large matrices, which is where configuration 4's doubled K buys the most.
    shapes = [
        (TM, TN, TK),
        (8, 16, 64),
        (32, 64, 128),
        (64, 3072, 1024),
        (240, 1024, 4096),
    ]
    for m, n, k in shapes:
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
        try:
            got = run(a, b)
        except Exception as error:
            check(f"{m}x{n}x{k}", False, str(error)[:110])
            continue
        want = a.astype(np.int32) @ b.astype(np.int32)
        wrong = int((got != want).sum())
        check(
            f"{m}x{n}x{k}",
            wrong == 0,
            "exact"
            if wrong == 0
            else f"{wrong} of {want.size} elements differ, worst {int(np.abs(got - want).max())}",
        )

    # Saturation: int8 x int8 over K=4096 can reach 4096*128*127, which is 6.7e7 and well
    # inside int32. The corner is worth pinning because it is the one an int16 accumulator
    # would fail and a reader may assume this one does too.
    m, n, k = 8, 16, 4096
    a = np.full((m, k), -128, np.int8)
    b = np.full((k, n), -128, np.int8)
    try:
        got = run(a, b)
        want = a.astype(np.int32) @ b.astype(np.int32)
        check(
            "saturation corner -128 x -128 over K=4096",
            bool((got == want).all()),
            f"{int(want[0, 0]):,} per element, int32 holds it",
        )
    except Exception as error:
        check("saturation corner -128 x -128 over K=4096", False, str(error)[:110])

    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        return 1
    print("gemm int8: configuration 4 is exact on every shape tried")
    return 0


if __name__ == "__main__":
    sys.exit(main())
