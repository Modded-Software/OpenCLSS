#!/usr/bin/env python3
"""Rank a frame's passes by *efficiency*, not just cost.

frame_profile answers "where did the time go" (ms, share). This answers the sharper
question: which passes do the *least work for the time they take*. For every recorded
call site it derives the arithmetic work from the shape in the label (FLOPs = 2*M*N*K per
batch) and divides by the measured per-call device time, giving GFLOP/s. Sorted ascending,
the top of the table is the worst offender - a pass that is slow *and* has almost nothing
to do, which is where a fix buys the most.

    NR_GPU_BACKEND=opencl XMX_DEVICE=1 uv run python src/bench/rank_ops.py [--size H W]
        [--runs N] [--seconds S] [--top N]

Non-GEMM sites (row, unary) carry no FLOP model here so they are listed separately with
their raw cost; the FLOP ranking is only meaningful for the GEMM-family shapes, which are
2/3 of the frame.
"""

import argparse
import re
import sys
import time

import frame_profile as fp
import nr_frame_resident as F
import nr_model
import numpy as np
from xmxrt import profile
from xmxrt.recorder import Recorder

# The first MxNxK (optionally " x<batch>") in a call-site label. Every GEMM-family label
# starts with its shape ("gemm 147456x128x32 flags 0x0", "ffn fused 640x256x64 x8 ...",
# "gemm+qkv epilogue 192x3072x1024"); non-GEMM labels (n=, rows=) have no such triple.
_SHAPE = re.compile(r"(\d+)x(\d+)x(\d+)(?:\s*x(\d+))?")


# Composed passes bundle several kernels under one label *and one timestamp span*, so the
# shape in the label is only the last of them. `ffn fused` is the expand (M x hidden x cin)
# plus the projection (M x 32 x hidden); for the stem/merge shapes cin=32, hidden=128 the two
# are equal, so the label's projection FLOPs double. Without this the ffn looks like the worst
# site in the frame when it is only ~average.
_COMPOSED = {"ffn fused": 2}


def _scale(label):
    for prefix, factor in _COMPOSED.items():
        if label.startswith(prefix):
            return factor
    return 1


def work_flops(label):
    m = _SHAPE.search(label)
    if not m:
        return None
    m_, n_, k_ = int(m[1]), int(m[2]), int(m[3])
    batch = int(m[4]) if m[4] else 1
    return 2 * m_ * n_ * k_ * batch * _scale(label)


def work_bytes(label):
    """Lower-bound device traffic for one call: A and B half, C float32. Deliberately
    ignores the epilogue's extra reads (gate/residual/skip), so the real GB/s is higher -
    this marks which shapes a *core-only* traffic model already calls I/O-bound."""
    m = _SHAPE.search(label)
    if not m:
        return None
    m_, n_, k_ = int(m[1]), int(m[2]), int(m[3])
    batch = int(m[4]) if m[4] else 1
    return (2 * (m_ * k_ + k_ * n_) + 4 * m_ * n_) * batch * _scale(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", nargs=2, type=int, default=(768, 1280))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    height, width = args.size

    model = nr_model.NeuralRenderingModel.from_safetensors(nr_model.default_weights())
    runtime = Recorder()
    profile.profile(True)
    frame = F.ResidentFrame(runtime, model.weights, height, width)
    features = (np.random.default_rng(11).standard_normal((height, width, 16)) * 0.3).astype(
        np.float32
    )
    frame.run(features, execution="single")
    until = time.perf_counter() + args.seconds
    while time.perf_counter() < until:
        frame.run(features, execution="single")

    log = fp.record_calls(runtime.lib)
    profile.profile_reset()
    for _ in range(args.runs):
        frame.run(features, execution="single")
    each = profile.profile_each()
    if len(each) != len(log):
        raise SystemExit(
            "%d passes timed against %d recorded: labels would not line up" % (len(each), len(log))
        )

    sites = {}
    for name, ms in zip(log, each, strict=False):
        total, count = sites.get(name, (0.0, 0))
        sites[name] = (total + max(ms, 0.0), count + 1)

    rows = []
    for name, (total, count) in sites.items():
        per_call = total / count
        flops = work_flops(name)
        nbytes = work_bytes(name)
        gflops = flops / per_call / 1e6 if flops and per_call > 0 else None
        gbps = nbytes / per_call / 1e6 if nbytes and per_call > 0 else None
        ai = flops / nbytes if flops and nbytes else None
        rows.append(
            (
                name,
                count / args.runs,
                per_call,
                total / args.runs,
                flops,
                gflops,
                nbytes,
                gbps,
                ai,
            )
        )

    ranked = [r for r in rows if r[5] is not None]
    ranked.sort(key=lambda r: r[5])  # worst (least work per time) first
    unknown = sorted((r for r in rows if r[5] is None), key=lambda r: -r[3])

    print(
        "%dx%d, %d runs - worst efficiency first (least FLOPs per ms)\n"
        % (width, height, args.runs)
    )
    print(
        "  %-44s %5s %8s %8s %8s %8s %7s %7s"
        % (
            "call site",
            "calls",
            "us each",
            "GFLOP/s",
            "ms tot",
            "GB/s",
            "FLOP/B",
            "waste%",
        )
    )
    best = max(r[5] for r in ranked) if ranked else 0.0
    for name, calls, per_call, total, flops, gflops, nbytes, gbps, ai in ranked[: args.top]:
        at_best = flops / best / 1e6 * calls if best else per_call * calls
        waste = 100 * (1 - at_best / total) if total else 0.0
        print(
            "  %-44s %5.0f %8.1f %8.1f %8.3f %8.1f %7.0f %6.0f%%"
            % (name[:44], calls, per_call * 1e3, gflops, total, gbps, ai, waste)
        )
    if unknown:
        print("\n  %-52s %6s %9s %9s" % ("(no FLOP model)", "calls", "us each", "ms total"))
        for name, calls, per_call, total, *_ in unknown[: args.top]:
            print("  %-52s %6.0f %9.1f %9.3f" % (name[:52], calls, per_call * 1e3, total))

    gemm_ms = sum(r[3] for r in ranked)
    gemm_work = sum(r[4] * r[1] for r in ranked)
    print(
        f"\n  GEMM-family: {gemm_work / 1e6:.1f} MFLOP over {gemm_ms:.2f} ms = {gemm_work / gemm_ms / 1e6:.1f} GFLOP/s average"
    )
    print(
        f"  best call site: {best:.1f} GFLOP/s; at that rate the same work would take {gemm_work / best / 1e6:.2f} ms"
    )


if __name__ == "__main__":
    sys.exit(main())
