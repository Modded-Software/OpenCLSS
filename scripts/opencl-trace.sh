#!/bin/bash
# Run one OpenCL frame twice with XMX_CL_TRACE (per-command output hashes, serialised
# queue) and diff the traces: the first differing line is the first command whose output
# changed run to run.
#   bash scripts/opencl-trace.sh [DUMP_LINE WIDTH]
# With DUMP_LINE, that trace line's buffers are dumped in both runs and compared as a
# WIDTH-wide matrix of 16-bit elements (frame_stages.py dumpdiff).
# Output: notes/runs/opencl-trace-{1,2}.txt[.LINE.role], notes/runs/opencl-trace.diff
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export UV_PROJECT_ENVIRONMENT=$PWD/.venv XMX_DEVICE=1 NR_GPU_BACKEND=opencl
LINE=${1:-0}
WIDTH=${2:-1}
for n in 1 2; do
  : > notes/runs/opencl-trace-$n.txt
  XMX_CL_DUMP_AT=$LINE XMX_CL_TRACE=$PWD/notes/runs/opencl-trace-$n.txt timeout -k 5 900 \
    uv run python src/gpu/frame_stages.py save notes/runs/stages-trace$n.npz \
    > notes/runs/opencl-trace-$n.out 2>&1
  echo "run $n exit $? lines $(wc -l < notes/runs/opencl-trace-$n.txt)"
done
diff notes/runs/opencl-trace-1.txt notes/runs/opencl-trace-2.txt > notes/runs/opencl-trace.diff
echo "diff lines $(wc -l < notes/runs/opencl-trace.diff)"
grep -m 4 -E "^[<>]" notes/runs/opencl-trace.diff
if [ "$LINE" != 0 ]; then
  grep "^# dumped" notes/runs/opencl-trace-1.txt
  for role in a b c d second; do
    f=notes/runs/opencl-trace-1.txt.$LINE.$role
    [ -f "$f" ] || continue
    echo "-- $role"
    timeout 120 uv run python src/gpu/frame_stages.py dumpdiff "$f" \
      notes/runs/opencl-trace-2.txt.$LINE.$role "$WIDTH"
  done
fi
