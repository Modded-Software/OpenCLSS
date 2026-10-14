#!/usr/bin/env python3
"""Every captured stage of one frame, saved, and compared against another backend's.

    NR_GPU_BACKEND=opencl uv run python src/gpu/frame_stages.py save notes/runs/stages-opencl.npz
    uv run python src/gpu/frame_stages.py save notes/runs/stages-vulkan.npz
    uv run python src/gpu/frame_stages.py compare notes/runs/stages-vulkan.npz notes/runs/stages-opencl.npz

The backends are bit-identical by contract, so the first stage in execution order that
differs is where a port's fault is. Capture runs one submit a block, so `save` twice on
the same backend also tells a nondeterministic stage apart.
"""

import sys

import numpy as np


def save(path, size=320):
    import nr_frame

    backend = nr_frame.ResidentBackend()
    try:
        color = np.random.default_rng(69).random((size, size, 3), dtype=np.float32)
        features = nr_frame.make_features(color, **nr_frame.PROFILES["standard"])
        frame = backend.frame(*features.shape[:2])
        capture = {}
        head = frame.run(features, capture=capture)
        order = list(capture)
        np.savez(path, __order__=np.array(order), __head__=head, **capture)
        print(f"saved {len(order)} stages to {path}")
    finally:
        backend.close()


def compare(want_path, got_path):
    want, got = np.load(want_path), np.load(got_path)
    first = None
    for name in [*list(want["__order__"]), "__head__"]:
        if name not in got:
            print(f"MISSING {name}")
            continue
        a, b = want[name], got[name]
        if a.shape != b.shape:
            print(f"SHAPE {name}: {a.shape} vs {b.shape}")
            first = first or name
            continue
        same = np.array_equal(
            a.view(np.uint32) if a.dtype == np.float32 else a,
            b.view(np.uint32) if b.dtype == np.float32 else b,
        )
        if same:
            print(f"same  {name}")
        else:
            diff = int((a != b).sum()) + int(np.isnan(a).sum() != np.isnan(b).sum())
            print(
                f"DIFF  {name}: {diff} of {a.size}, max |d| {np.nanmax(np.abs(a - b)):.4g}, "
                f"nan {int(np.isnan(b).sum())}"
            )
            first = first or name
    print(f"first difference: {first}" if first else "all stages identical")
    return 0 if first is None else 1


def dumpdiff(first, second, width, itemsize=2):
    """Two XMX_CL_DUMP_AT dumps of one buffer: where they differ, as (row, column) of a
    `width`-wide matrix of `itemsize`-byte elements."""
    kind = np.uint16 if itemsize == 2 else np.uint32
    a, b = np.fromfile(first, kind), np.fromfile(second, kind)
    where = np.flatnonzero(a != b)
    print(f"{where.size} of {a.size} elements differ")
    if where.size:
        rows, cols = where // width, where % width
        print("rows", np.unique(rows)[:40].tolist(), "count", np.unique(rows).size)
        print("cols", np.unique(cols)[:40].tolist(), "count", np.unique(cols).size)
        print("first/last index", int(where[0]), int(where[-1]))
        if itemsize == 2:
            sample = where[:8]
            print(
                "values",
                list(
                    zip(
                        a[sample].view(np.float16).tolist(),
                        b[sample].view(np.float16).tolist(),
                        strict=False,
                    )
                ),
            )
    return 0 if where.size == 0 else 1


if __name__ == "__main__":
    if sys.argv[1] == "save":
        save(sys.argv[2])
    elif sys.argv[1] == "dumpdiff":
        sys.exit(
            dumpdiff(
                sys.argv[2],
                sys.argv[3],
                int(sys.argv[4]),
                int(sys.argv[5]) if len(sys.argv) > 5 else 2,
            )
        )
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
