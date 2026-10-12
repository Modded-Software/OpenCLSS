#!/bin/bash
# Build everything for the Linux/Vulkan (Arc DG2) path with the repo-local deps in work/deps
# (glslang, X11 and libpng headers). Make variables pass through the environment:
#   STAGED_SG=8 FFN_SG=16 WINDOW_SG=16 STAGED_WIDE_SUBS_M=8 STAGED_WIDE_WARPS=8
# Prints MAKE_DONE on success.
set -e
R=${NR_ROOT:-$PWD}   # the repo root; $0 is not a path under every launcher
[ -f "$R/Makefile" ] || { echo "run from refs/dlss-on-arc or set NR_ROOT"; exit 2; }
DEPS=$R/work/deps
if [ ! -f "$DEPS/x11/usr/include/X11/Xlib.h" ]; then
  mkdir -p "$DEPS" && cd "$DEPS"
  apt-get download libx11-dev x11proto-dev
  for d in libx11-dev_*.deb x11proto-dev_*.deb; do dpkg -x "$d" x11; done
fi
cd "$R"
export PATH=$DEPS/glslang/bin:/usr/bin:/bin:$PATH
export CPATH=$DEPS/x11/usr/include
P=$DEPS/png/usr
rm -f work/nr_frame.o
make -j16 LAYER32=0 CC=/usr/bin/gcc \
  PNG_CFLAGS="-I$P/include/libpng16" \
  PNG_LIBS="$P/lib/x86_64-linux-gnu/libpng16.a -lz"
echo MAKE_DONE
