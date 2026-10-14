#!/usr/bin/env python3
"""A before/after crop, side by side, to look at what the network actually did."""

import sys

import image_io
import numpy as np

before = image_io.load(sys.argv[1])
after = image_io.load(sys.argv[2])
y, x, size = (int(v) for v in sys.argv[4:7]) if len(sys.argv) > 6 else (300, 560, 260)
a = before[y : y + size, x : x + size]
b = after[y : y + size, x : x + size]
gap = np.ones((size, 8, 3), np.float32)
image_io.save(np.concatenate([a, gap, b], axis=1), sys.argv[3])
print(f"wrote {sys.argv[3]}  (left: input, right: neurally rendered)")
print(f"  crop sd  {a.std():.4f} -> {b.std():.4f}   mean|difference| {np.abs(a - b).mean():.4f}")
