#!/usr/bin/env python3
"""Sustained throughput of the resident GEMM - the kernel the frame graph dispatches.

Each shape is recorded as one command buffer of back-to-back GEMMs (the graph's barrier
between them, as in a frame) and submitted repeatedly for a wall-clock budget after an
equal warm-up, so the GPU clock is at its loaded level: no 10 ms samples. The rate is over
the whole sustained interval. Every shape is checked against NumPy first, so a fast wrong
kernel cannot be reported.

    python3 src/bench/gemm_sustained.py                       # the frame's hot shapes, 2 s each
    python3 src/bench/gemm_sustained.py 640x512x512 --seconds 5 --record baseline

Shapes are MxNxK (A MxK half, B KxN half, C MxN float32). `--record` appends one JSON
line per shape to notes/perf-log.jsonl.
"""

import argparse
import json
import pathlib
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
from xmxrt import gemm, memory
from xmxrt.recorder import Recorder

# the frame's hottest staged GEMMs at 1024x576 (frame_profile.py --calls), as plain GEMMs
HOT = [
    "192x1024x4096",
    "960x1536x512",
    "2304x128x256",
    "9216x128x128",
    "36864x128x64",
    "192x3072x1024",
    "640x512x512",
    "192x4096x1024",
    "36864x32x128",
    "4096x1024x4096",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("shapes", nargs="*", default=HOT)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--record", metavar="LABEL")
    args = parser.parse_args()
    rt = Recorder()
    rng = np.random.default_rng(0)
    version = subprocess.run(
        ["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
    )
    if version.returncode != 0:
        raise RuntimeError(f"git describe failed: {version.stderr.strip()}")
    commit = version.stdout.strip()
    print("  %-18s %10s %10s %8s %9s" % ("shape", "us/gemm", "TFLOP/s", "gemms", "rel err"))
    worst = 0.0
    for shape in args.shapes:
        M, N, K = (int(x) for x in shape.split("x"))
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
        a, b, c = (
            rt.buffer(M * K, np.float16),
            rt.buffer(K * N, np.float16),
            rt.buffer(M * N),
        )
        memory.host_write(a, A)
        memory.host_write(b, B)
        # correctness, once
        rt.begin()
        gemm.gemm(rt, a, b, c, M, N, K)
        rt.submit()
        ref = A.astype(np.float32) @ B.astype(np.float32)
        err = float(
            np.abs(memory.host_view(c)[: M * N].reshape(M, N) - ref).max()
            / max(np.abs(ref).max(), 1e-30)
        )
        worst = max(worst, err)
        # one captured command buffer of ~25 ms of GEMMs (sized from a probe), replayed:
        # no Python recording inside the timed loop
        rt.begin()
        gemm.gemm(rt, a, b, c, M, N, K)
        t = time.perf_counter()
        rt.submit()
        per = max(time.perf_counter() - t, 1e-5)
        reps = max(1, min(4000, int(0.025 / per)))
        rt.begin()
        for _ in range(reps):
            gemm.gemm(rt, a, b, c, M, N, K)
        graph = rt.lib.xmx_graph_capture()
        if graph < 0:
            raise SystemExit("xmx_graph_capture: " + rt.lib.xmx_error().decode())
        until = time.perf_counter() + args.seconds  # warm-up at load
        while time.perf_counter() < until:
            rt.lib.xmx_graph_run(graph)
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            if rt.lib.xmx_graph_run(graph) < 0:
                raise SystemExit("xmx_graph_run: " + rt.lib.xmx_error().decode())
            n += reps
        dt = time.perf_counter() - t0
        rt.lib.xmx_graph_destroy(graph)
        tflops = 2.0 * M * N * K * n / dt / 1e12
        print(
            "  %-18s %10.1f %10.2f %8d %9.1e" % (shape, 1e6 * dt / n, tflops, n, err),
            flush=True,
        )
        if args.record:
            with open(ROOT / "notes" / "perf-log.jsonl", "a") as sink:
                sink.write(
                    json.dumps(
                        {
                            "label": args.record,
                            "kind": "gemm",
                            "commit": commit,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "shape": shape,
                            "us": round(1e6 * dt / n, 2),
                            "tflops": round(tflops, 3),
                            "seconds": round(dt, 2),
                            "err": err,
                        }
                    )
                    + "\n"
                )
        for buf in (a, b, c):
            buf.free()
    raise SystemExit(0 if worst < 1e-2 else 1)


if __name__ == "__main__":
    main()
