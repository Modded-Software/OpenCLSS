#!/bin/bash
# DPAS register-only peak vs independent accumulator chains and subgroup width.
# CHAINS=8 spills on DG2 SIMD16 (44:18 spills:fills, INTEL_DEBUG=cs), so the default probe
# understates the ceiling. Appends to notes/runs/coopmat-peak-chains.log.
R=${NR_ROOT:-$PWD}
export PATH=$R/work/deps/glslang/bin:/usr/bin:/bin:$PATH
cd "$R" || exit 2
mkdir -p notes/runs
/usr/bin/gcc -O2 -Iwork/vulkan-headers/include -o work/coopmat_peak src/probe/coopmat_peak.c -lvulkan || exit 1
LOG=notes/runs/coopmat-peak-chains.log
date -Is >> $LOG
for c in ${CHAINS_LIST:-1 2 4 6 8}; do
  glslangValidator --target-env vulkan1.3 -DXMX_TN=8 -DCHAINS=$c -o work/coopmat_peak_c$c.spv src/probe/coopmat_peak.comp > work/coopmat_peak_c$c.log 2>&1 || { cat work/coopmat_peak_c$c.log; exit 1; }
  for sg in 8 16 32; do
    echo "chains=$c $(work/coopmat_peak work/coopmat_peak_c$c.spv $sg 1 8 $c)" >> $LOG
  done
done
cat $LOG
