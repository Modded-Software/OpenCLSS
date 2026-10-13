#!/usr/bin/env python3
"""Every pass, run twice on the same inputs, must write the same bytes.

A backend race shows up here as a run-to-run difference in one stage's output, before
it is lost among the comparisons of a fused pass against its reference. Each stage of
the global attention chain is run in its own submit and checked on its own, over shapes
whose extents are and are not whole 32- and 64-blocks.
"""
import numpy as np
import xmxres as X

SHAPES = ((48, 48, 4), (96, 90, 8), (112, 105, 4), (112, 112, 4), (128, 128, 4),
          (80, 80, 2), (144, 144, 2), (240, 240, 2))
REPEATS = 3


def stages(rt, rows, tokens, heads, rng):
    ch = heads * 32
    q, k, v = (rt.buffer(heads * rows * 32, np.float16) for _ in range(3))
    scores = rt.buffer(heads * rows * rows, np.float32)
    probs = rt.buffer(heads * rows * rows, np.float16)
    recip = rt.buffer(heads * rows, np.float32)
    ctx = rt.buffer(heads * rows * 32, np.float32)
    merged = rt.buffer(rows * ch, np.float16)
    for b in (q, k, v):
        X.host_write(b, rng.normal(0, .2, heads * rows * 32).astype(np.float16))
    run = [
        ("qk", scores, np.float32, lambda: rt.gemm(
            q, k, scores, rows, rows, 32, batch=heads,
            strides=(rows * 32, rows * 32, rows * rows), transpose_b=True)),
        # the same product with K stored [32][rows], no transposed read: tells a transposed
        # load's fault from the multiply's
        ("qk_nt", scores, np.float32, lambda: rt.gemm(
            q, k, scores, rows, rows, 32, batch=heads,
            strides=(rows * 32, rows * 32, rows * rows))),
        ("qk", scores, np.float32, lambda: rt.gemm(
            q, k, scores, rows, rows, 32, batch=heads,
            strides=(rows * 32, rows * 32, rows * rows), transpose_b=True)),
        ("softmax", probs, np.float16, lambda: rt.vit_softmax(
            scores, probs, recip, heads * rows, tokens, stride=rows)),
        ("pv", ctx, np.float32, lambda: rt.gemm(
            probs, v, ctx, rows, 32, rows, batch=heads,
            strides=(rows * rows, rows * 32, rows * 32))),
        ("merge", merged, np.float16, lambda: rt.merge_heads(
            ctx, merged, 1, rows, ch, heads, epilogue=X.EPI_E4M3, narrow=True, scale=recip)),
    ]
    bad = []
    for name, out, dtype, record in run:
        seen = None
        for _ in range(REPEATS):
            rt.begin()
            record()
            rt.submit()
            got = X.host_view(out, dtype).copy().view(np.uint16 if dtype == np.float16 else np.uint32)
            if seen is None:
                seen = got
            elif not np.array_equal(seen, got):
                where = np.flatnonzero(seen != got)
                width = rows if name in ("qk", "softmax") else 32 if name == "pv" else ch
                spots = ", ".join(f"(b{i // (rows * width)} r{i // width % rows} c{i % width}"
                                  f" {seen.view(dtype)[i]:.4g}/{got.view(dtype)[i]:.4g})"
                                  for i in where[:6])
                bad.append(f"{name}: {where.size} of {got.size} differ: {spots}")
                break
    for b in (q, k, v, scores, probs, recip, ctx, merged):
        b.free()
    return bad


# The frame's projection GEMMs at each encoder level's (tokens, C): qkv, out, ffn up/down.
# Encoder L4 (40x40, C=256 at 320) was the first frame stage to differ run to run.
PROJECTIONS = tuple((t, n, k) for t, c in ((6400, 128), (1600, 256), (400, 512))
                    for n, k in ((3 * c, c), (c, c), (4 * c, c), (c, 4 * c)))


def projections(rt, rng):
    bad = []
    for m, n, k in PROJECTIONS:
        a, b = rt.buffer(m * k, np.float16), rt.buffer(k * n, np.float16)
        c = rt.buffer(m * n, np.float32)
        X.host_write(a, rng.normal(0, .2, m * k).astype(np.float16))
        X.host_write(b, rng.normal(0, .2, k * n).astype(np.float16))
        seen = None
        for _ in range(REPEATS):
            rt.begin()
            rt.gemm(a, b, c, m, n, k)
            rt.submit()
            got = X.host_view(c, np.float32).copy().view(np.uint32)
            if seen is None:
                seen = got
            elif not np.array_equal(seen, got):
                where = np.flatnonzero(seen != got)
                bad.append(f"gemm M={m} N={n} K={k}: {where.size} of {got.size} differ, "
                           f"first rows {sorted(set((where // n)[:8].tolist()))}")
                break
        for buf in (a, b, c):
            buf.free()
    return bad


# (windows, heads) of the encoder levels at 320: L3 100x4, L4 25x8, L5 ~9x16.
WINDOWS = ((100, 4), (25, 8), (9, 16))


def windows(rt, rng):
    bad = []
    for count, heads in WINDOWS:
        batches = count * heads
        q, k, v = (rt.buffer(batches * 2048, np.float16) for _ in range(3))
        bias = rt.buffer(heads * 4096, np.float32)
        for b in (q, k, v):
            X.host_write(b, rng.normal(0, .3, batches * 2048).astype(np.float16))
        X.host_write(bias, rng.normal(0, .5, heads * 4096).astype(np.float32))
        for merged in (False, True):
            out = rt.buffer(batches * 2048 if merged else batches * 2048, np.float16 if merged
                            else np.float32)
            dtype = np.float16 if merged else np.float32
            seen = None
            for _ in range(REPEATS):
                rt.begin()
                rt.window_attention(q, k, v, out, batches, heads, bias=bias, merged=merged)
                rt.submit()
                got = X.host_view(out, dtype).copy().view(np.uint16 if merged else np.uint32)
                if seen is None:
                    seen = got
                elif not np.array_equal(seen, got):
                    where = np.flatnonzero(seen != got)
                    bad.append(f"window_attention windows={count} heads={heads} merged={merged}: "
                               f"{where.size} of {got.size} differ, first batches "
                               f"{sorted(set((where // 2048)[:8].tolist()))}")
                    break
            out.free()
        for b in (q, k, v, bias):
            b.free()
    return bad


# The feed-forward's closing GEMM with the residual, half skip and E4M3 half output
# (bt=0x61100): the first frame command to differ run to run (scripts/opencl-trace.sh).
RESIDUALS = ((6400, 64, 64), (1600, 128, 128), (400, 256, 256), (144, 512, 512),
             (400, 256, 1024), (1600, 128, 512))


def residuals(rt, rng):
    bad = []
    for m, n, k in RESIDUALS:
        a, b = rt.buffer(m * k, np.float16), rt.buffer(k * n, np.float16)
        skip, cos = rt.buffer(m * n, np.float16), rt.buffer(n, np.float32)
        out = rt.buffer(m * n, np.float16)
        X.host_write(a, rng.normal(0, .2, m * k).astype(np.float16))
        X.host_write(b, rng.normal(0, .2, k * n).astype(np.float16))
        X.host_write(skip, rng.normal(0, 1, m * n).astype(np.float16))
        X.host_write(cos, rng.normal(0, 1, n).astype(np.float32))
        seen = None
        for _ in range(REPEATS):
            rt.begin()
            rt.gemm_residual(a, b, skip, cos, out, m, n, k, epilogue=X.EPI_E4M3, narrow=True,
                             skip_half=True)
            rt.submit()
            got = X.host_view(out, np.float16).copy().view(np.uint16)
            if seen is None:
                seen = got
            elif not np.array_equal(seen, got):
                where = np.flatnonzero(seen != got)
                bad.append(f"gemm_residual M={m} N={n} K={k}: {where.size} of {got.size} "
                           f"differ, rows {sorted(set((where // n).tolist()))[:12]} cols "
                           f"{sorted(set((where % n).tolist()))[:12]}")
                break
        for buf in (a, b, skip, cos, out):
            buf.free()
    return bad


def main():
    rt = X.Runtime()
    rng = np.random.default_rng(77)
    failures = residuals(rt, rng) + projections(rt, rng) + windows(rt, rng)
    for rows, tokens, heads in SHAPES:
        for problem in stages(rt, rows, tokens, heads, rng):
            failures.append(f"rows={rows} tokens={tokens} heads={heads} {problem}")
    for line in failures:
        print("NONDETERMINISTIC", line)
    assert not failures, f"{len(failures)} nondeterministic stages"
    print(f"determinism: {len(SHAPES)} shapes x 4 stages x {REPEATS} runs identical")


if __name__ == "__main__":
    main()
