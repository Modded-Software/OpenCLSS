#!/usr/bin/env python3
"""A whole frame, run twice in each execution mode, against itself and across modes.

Tells a nondeterministic pass (one mode differing from itself) from a mode that records
differently (modes self-consistent, but not equal to each other).
"""

import numpy as np


def main():
    import nr_frame

    if not nr_frame.WEIGHTS.exists():
        print("frame determinism: skipped (no logical weights)")
        return
    backend = nr_frame.ResidentBackend()
    problems = []
    try:
        color = np.random.default_rng(69).random((320, 320, 3), dtype=np.float32)
        features = nr_frame.make_features(color, **nr_frame.PROFILES["standard"])
        frame = backend.frame(*features.shape[:2])
        heads = {}
        for execution in ("single", "block", "replay"):
            runs = [frame.run(features, execution=execution).copy() for _ in range(2)]
            if not np.array_equal(runs[0], runs[1], equal_nan=True):
                bad = int((runs[0] != runs[1]).sum())
                problems.append(f"{execution}: differs from itself in {bad} values")
            heads[execution] = runs[0]
            finite = float(np.isfinite(runs[0]).mean())
            print(f"{execution}: finite {finite:.4f}, mean {np.nanmean(runs[0]):.5f}")
        for other in ("block", "replay"):
            if not np.array_equal(heads["single"], heads[other], equal_nan=True):
                bad = int((heads["single"] != heads[other]).sum())
                problems.append(f"single vs {other}: {bad} values differ")
    finally:
        backend.close()
    for p in problems:
        print("MISMATCH", p)
    assert not problems, f"{len(problems)} mismatches"
    print("frame determinism: every mode equal to itself and to the others")


if __name__ == "__main__":
    main()
