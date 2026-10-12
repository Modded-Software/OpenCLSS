#!/bin/bash
# Register-only DPAS peak per subgroup width (src/probe/coopmat_peak): the ceiling the
# GEMM kernels are measured against. Output also appended to notes/runs/coopmat-peak.log.
R=${NR_ROOT:-$PWD}   # the repo root; $0 is not a path under every launcher
export PATH=$R/work/deps/glslang/bin:/usr/bin:/bin:$PATH
cd "$R"
mkdir -p notes/runs
glslangValidator --target-env vulkan1.3 -DXMX_TN=8 -o work/coopmat_peak_n8.spv src/probe/coopmat_peak.comp
/usr/bin/gcc -O2 -Iwork/vulkan-headers/include -o work/coopmat_peak src/probe/coopmat_peak.c -lvulkan
{ date -Is; for sg in 8 16 32; do work/coopmat_peak work/coopmat_peak_n8.spv $sg 2; done; } \
  | tee -a notes/runs/coopmat-peak.log
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "coopmat-peak failed" >&2; exit 1; }
