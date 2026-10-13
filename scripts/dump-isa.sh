#!/bin/bash
# Dump Mesa's compiled ISA for every pipeline a gemm_sustained run builds:
#   scripts/dump-isa.sh OUT.txt [SHAPE]      (default 4096x1024x4096)
# The shader cache is disabled so every pipeline is really compiled (and printed).
R=${NR_ROOT:-$PWD}
cd "$R" || exit 2
OUT=${1:?usage: $0 OUT.txt [SHAPE]}
export UV_PROJECT_ENVIRONMENT=$R/.venv NR_BUILD_DIR=$R/work XMX_DEVICE=${XMX_DEVICE:-1}
export MESA_SHADER_CACHE_DISABLE=1 INTEL_DEBUG=cs
uv run python src/bench/gemm_sustained.py \
  --seconds 0.3 "${2:-4096x1024x4096}" > "$OUT" 2>&1
grep -n "SIMD.*shader:" "$OUT"
