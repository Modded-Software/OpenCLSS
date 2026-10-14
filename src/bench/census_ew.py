#!/usr/bin/env python3
"""Tally the elementwise / row passes one frame records, by kind."""

import collections
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
import nr_frame_resident as F
import nr_model
from xmxrt import attention as attention_module
from xmxrt import elementwise as elementwise_module
from xmxrt.recorder import Recorder

NAME = {
    0: "e4m3",
    1: "gate",
    2: "half",
    3: "to_half",
    4: "scale",
    5: "residual",
    6: "from_half",
    7: "partition",
    8: "reverse",
    9: "add_bias",
    10: "split_heads",
    11: "merge_heads",
    12: "pool2",
    13: "upsample2",
    14: "scale_channel",
    15: "add",
    16: "pad_end",
    17: "gate_e4m3_half",
    18: "e4m3_half",
    19: "gate_half",
}
ELEM, CALLS = collections.Counter(), collections.Counter()
real_unary = elementwise_module.unary
real_cos, real_soft = attention_module.cosine_publish, attention_module.softmax


def unary(runtime, kind, source, target, count, **kw):
    ELEM[NAME.get(kind, kind)] += count
    CALLS[NAME.get(kind, kind)] += 1
    return real_unary(runtime, kind, source, target, count, **kw)


def cosine(runtime, source, target, rows, **kw):
    ELEM["cosine_publish"] += rows * 32
    CALLS["cosine_publish"] += 1
    return real_cos(runtime, source, target, rows, **kw)


def softmax(runtime, source, target, rows, width, **kw):
    ELEM["softmax"] += rows * width
    CALLS["softmax"] += 1
    return real_soft(runtime, source, target, rows, width, **kw)


elementwise_module.unary = unary
attention_module.cosine_publish, attention_module.softmax = cosine, softmax

H, W = 768, 1280
model = nr_model.NeuralRenderingModel.from_safetensors(
    ROOT / "work" / "mlxw" / "dlssnr-logical.safetensors"
)
rt = Recorder()
F.ResidentFrame(rt, model.weights, H, W).run(np.zeros((H, W, 16), dtype=np.float32))

print("  %-18s %8s %14s %12s" % ("pass", "calls", "elements", "M elem"))
for name, count in ELEM.most_common():
    print("  %-18s %8d %14d %12.1f" % (name, CALLS[name], count, count / 1e6))
print(
    "  %-18s %8d %14d %12.1f"
    % ("TOTAL", sum(CALLS.values()), sum(ELEM.values()), sum(ELEM.values()) / 1e6)
)
