#!/bin/bash
# Run OpenCL gate tests synchronously, each under a hard timeout, and print one line per
# test plus its first failure line. Never waits on anything unbounded.
#   bash scripts/opencl-test.sh [--build] [--timeout SECS] [test_name ...]
# No names: every src/gpu/test_*.py. --build rebuilds only work/libopenclmx.so.
# Per-test logs: notes/runs/opencl-test/<test>.log; summary: notes/runs/opencl-test.log
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
LIMIT=180
BUILD=0
TESTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1 ;;
    --timeout) shift; LIMIT="$1" ;;
    *) TESTS+=("$1") ;;
  esac
  shift
done
DIR=notes/runs/opencl-test
SUM=notes/runs/opencl-test.log
mkdir -p "$DIR"
: > "$SUM"
# one line to the summary and the console, without piping into tee (the CI hook rejects a
# pipe whose exit status is not checked)
say() { printf '%s\n' "$*" >> "$SUM"; printf '%s\n' "$*"; }
if [ "$BUILD" = 1 ]; then
  # the system linker: the workspace's own ld cannot load gcc's LTO plugin
  if ! PATH=/usr/bin:/bin:$PATH timeout 300 make CC=/usr/bin/gcc work/libopenclmx.so > "$DIR/build.log" 2>&1; then
    say "BUILD FAILED"
    tail -n 40 "$DIR/build.log" >> "$SUM"
    tail -n 40 "$DIR/build.log"
    exit 1
  fi
  say "BUILD OK"
fi
if [ ${#TESTS[@]} -eq 0 ]; then
  for f in src/gpu/test_*.py; do TESTS+=("$(basename "$f" .py)"); done
fi
PASS=0
FAIL=0
for t in "${TESTS[@]}"; do
  LOG="$DIR/$t.log"
  NR_GPU_BACKEND=opencl XMX_DEVICE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
    timeout -k 5 "$LIMIT" uv run python "src/gpu/$t.py" > "$LOG" 2>&1
  STATUS=$?
  if [ "$STATUS" = 0 ]; then
    PASS=$((PASS + 1))
    say "PASS $t"
  else
    FAIL=$((FAIL + 1))
    if [ "$STATUS" = 124 ] || [ "$STATUS" = 137 ]; then WHY="TIMEOUT after ${LIMIT}s"
    else WHY="$(grep -m 1 -E "^[A-Za-z]*Error|not implemented" "$LOG")"; fi
    say "FAIL $t: $WHY"
  fi
done
say "TOTAL pass=$PASS fail=$FAIL"
