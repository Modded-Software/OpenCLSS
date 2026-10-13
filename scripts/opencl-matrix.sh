#!/usr/bin/env bash
# Runs every src/gpu test under NR_GPU_BACKEND=opencl; one PASS/FAIL line per test,
# full output per test in notes/runs/opencl-matrix/. Summary: notes/runs/opencl-matrix.log
# Run from the repo root (the shell guard hands scripts in as /proc/self/fd/N, so $0 is no path).
export UV_PROJECT_ENVIRONMENT=$PWD/.venv NR_GPU_BACKEND=opencl XMX_DEVICE=${XMX_DEVICE:-1}
out=notes/runs/opencl-matrix
mkdir -p "$out"
: > "$out.log"
for t in src/gpu/test_*.py; do
	n=$(basename "$t" .py)
	if timeout 600 uv run python "$t" > "$out/$n.log" 2>&1; then
		echo "PASS $n" >> "$out.log"
	else
		echo "FAIL $n" >> "$out.log"
	fi
done
echo "DONE END" >> "$out.log"
