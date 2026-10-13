#!/bin/bash
# The block-read OpenCL GEMM against the scalar one, same shapes, checked against numpy:
#   scripts/opencl-blk.sh [LOG]       (default notes/runs/opencl-blk.log)
R=${NR_ROOT:-$PWD}
cd "$R" || exit 2
LOG=${1:-notes/runs/opencl-blk.log}
export UV_PROJECT_ENVIRONMENT=$R/.venv
: > "$LOG"
for shape in 4096,1024,4096 192,4096,1024 960,1536,512 9216,128,128 36864,128,64; do
	for src in opencl_gemm8.cl opencl_gemm8_blk.cl; do
		for ku in 1 2; do
			echo "== $src $shape kunroll $ku" >> "$LOG"
			CL_SRC=$src uv run python src/bench/opencl_gemm8.py --device 1 --check \
				--shape "$shape" --kunroll "$ku" --blocks 2x2,4x2,2x4,4x4,8x2,8x4 >> "$LOG" 2>&1
		done
	done
done
grep -c "FAILED\|Traceback\|Error" "$LOG"
