#!/bin/bash
# Capture one frame's stages on Vulkan and OpenCL (each with a hard timeout) and print
# the comparison: the first stage that differs is where the OpenCL port is wrong.
#   bash scripts/frame-stages.sh [--opencl-twice]
# --opencl-twice compares two OpenCL captures with each other: a nondeterministic stage.
# Output: notes/runs/frame-stages.log; the captures: notes/runs/stages-*.npz
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
OUT=notes/runs/frame-stages.log
export UV_PROJECT_ENVIRONMENT=$PWD/.venv XMX_DEVICE=1
: > "$OUT"
if [ "${1:-}" = "--opencl-twice" ]; then
  for n in 1 2; do
    NR_GPU_BACKEND=opencl timeout -k 5 600 uv run python src/gpu/frame_stages.py save notes/runs/stages-opencl$n.npz >> "$OUT" 2>&1
    echo "opencl save $n exit $?" >> "$OUT"
  done
  timeout 120 uv run python src/gpu/frame_stages.py compare notes/runs/stages-opencl1.npz notes/runs/stages-opencl2.npz >> "$OUT" 2>&1
  echo "compare exit $?" >> "$OUT"
  grep -E "exit|DIFF|SHAPE|MISSING|first difference|all stages|Error" "$OUT"
  exit 0
fi
if true; then
  timeout -k 5 600 uv run python src/gpu/frame_stages.py save notes/runs/stages-vulkan.npz >> "$OUT" 2>&1
  echo "vulkan save exit $?" >> "$OUT"
fi
NR_GPU_BACKEND=opencl timeout -k 5 600 uv run python src/gpu/frame_stages.py save notes/runs/stages-opencl.npz >> "$OUT" 2>&1
echo "opencl save exit $?" >> "$OUT"
timeout 120 uv run python src/gpu/frame_stages.py compare notes/runs/stages-vulkan.npz notes/runs/stages-opencl.npz >> "$OUT" 2>&1
echo "compare exit $?" >> "$OUT"
grep -E "exit|DIFF|SHAPE|MISSING|first difference|all stages|Error" "$OUT"
