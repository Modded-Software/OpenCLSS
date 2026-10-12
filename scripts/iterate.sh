#!/bin/bash
# One optimisation iteration, everything persisted in the repo:
#   scripts/iterate.sh LABEL [iteration.py args]     (Make knobs via env, see dg2-build.sh)
# Clean-rebuilds the subgroup-pinned kernels, builds, then src/bench/iteration.py LABEL:
# tests + sustained benches + pipeline stats -> notes/runs.jsonl, notes/perf-log.jsonl and
# notes/runs/LABEL/ (build log, test and bench output, pipeline stats). Prints the summary
# and the comparison table. Exit status is the iteration's.
# run from the repo root (or set NR_ROOT): $0 is not a path under every launcher
R=${NR_ROOT:-$PWD}
[ -f "$R/src/bench/iteration.py" ] || { echo "run from refs/dlss-on-arc or set NR_ROOT"; exit 2; }
LABEL=$1
[ -n "$LABEL" ] || { echo "usage: $0 LABEL [args]"; exit 2; }
OUT=$R/notes/runs/$LABEL
mkdir -p "$OUT"
# GPU 0 shares the box with everything else; test on GPU 1 unless told otherwise
export XMX_DEVICE=${XMX_DEVICE:-1}
env | grep -E '^(STAGED_SG|FFN_SG|FFN_ROWS|WINDOW_SG|STAGED_WIDE_[A-Z_]+|XMX_[A-Z_]+)=' > "$OUT/env.txt"
rm -f "$R"/work/gemm_staged*_n8.spv "$R"/work/ffn_fused_n8.spv "$R"/work/window_block_n8.spv \
      "$R"/work/window_attention_n8.spv "$R"/work/libxmx.so "$R"/work/libnr_frame.so
bash "$R/scripts/dg2-build.sh" > "$OUT/build.log" 2>&1
grep -q MAKE_DONE "$OUT/build.log" || { echo BUILD_FAILED; grep -n error "$OUT/build.log"; tail -n 40 "$OUT/build.log"; exit 1; }
cd "$R"
export VIRTUAL_ENV=$R/.venv NR_BUILD_DIR=$R/work
uv run --active python src/bench/iteration.py "$@" > "$OUT/iteration.json"
STATUS=$?
uv run --active python src/bench/iteration.py --summary "$LABEL"
uv run --active python src/bench/iteration.py --compare
exit $STATUS
