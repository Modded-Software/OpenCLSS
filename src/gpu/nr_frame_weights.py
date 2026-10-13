#!/usr/bin/env python3
"""DeviceWeights - every weight buffer, and the edges that join the blocks.

Split out of `nr_frame_resident`: uploading the weights knows nothing about the extent, so
it need not live with the graph that does. They used to be the frame's, and a frame belongs
to one extent - so moving the render scale rebuilt them, ~292 MB written again for a knob.
Nothing in them knows the extent: blocks are keyed by index, edges by index and direction.
So one set lives per backend and a new extent rebuilds the graph and its scratch alone.
"""
from __future__ import annotations

import numpy as np

import nr_resident as R

# the named weight buffers: a frame forwards attribute reads to the DeviceWeights it holds
WEIGHT_NAMES = ("adapter", "bottleneck", "decoder_input", "merge_sin", "merge_cos",
                "merge_sincos", "head")


class Edge:
    """A transition's weights: a projection and, for the decoder, a skip scale."""

    def __init__(self, runtime, weight, sine=None):
        self.weight0 = runtime.buffer_from(weight, np.float16)
        self.out_channels = weight.shape[1]
        self.sine = None if sine is None else runtime.buffer_from(sine)


class DeviceWeights:
    """Every weight buffer, and nothing that depends on the extent."""

    def __init__(self, runtime, weights):
        self.rt, self.weights = runtime, weights
        self._blocks, self._edges = {}, {}
        self._upload_edges()

    def _upload_edges(self):
        take = lambda name: self.weights[name]
        self.adapter = self.rt.buffer_from(take("block0.layer0.input_adapter_weight"),
                                           np.float16)
        self.bottleneck = Edge(self.rt, take("block30.layer4.weight"))
        self.decoder_input = Edge(self.rt, take("block39.layer0.conv_weight"),
                                  take("block39.layer0.inp_upsample_sin"))
        self.merge_sin = self.rt.buffer_from(take("block70.layer0.inp_merge_sin"))
        self.merge_cos = self.rt.buffer_from(take("block70.layer0.inp_merge_cos"))
        # both, one after the other, for the pass that applies them together: a prepared
        # file carries them joined (nr_frame_prepare), else they are joined here
        sincos = self.weights.get("block70.layer0.inp_merge_sincos")
        if sincos is None:
            sincos = np.concatenate([
                np.asarray(take("block70.layer0.inp_merge_sin"), np.float32).reshape(-1),
                np.asarray(take("block70.layer0.inp_merge_cos"), np.float32).reshape(-1)])
        self.merge_sincos = self.rt.buffer_from(sincos)
        # the head is 32 -> 4, and the cooperative matrix wants a multiple of 16
        # columns; both halves go into one padded matrix and the first four columns
        # of the product are the head - a prepared file carries the matrix
        head = self.weights.get("block70.layer0.head_matrix")
        if head is None:
            head = np.zeros((32, 16), dtype=np.float32)
            head[:16, :4] = take("block70.layer0.out_gain")
            head[16:, :4] = take("block70.layer0.out_conv_weight")
        self.head = self.rt.buffer_from(head, np.float16)

    def block(self, index, heads, family="window"):
        if (index, family) not in self._blocks:
            builder = {"split": lambda: R.SplitBlockWeights(self.rt, self.weights, index),
                       "global": lambda: R.GlobalBlockWeights(self.rt, self.weights, index),
                       "window": lambda: R.BlockWeights(self.rt, self.weights, index,
                                                        heads=heads)}[family]
            self._blocks[(index, family)] = builder()
        return self._blocks[(index, family)]

    def edge(self, index, kind):
        if (index, kind) not in self._edges:
            prefix = f"block{index}.layer0"
            self._edges[(index, kind)] = Edge(
                self.rt, self.weights[f"{prefix}.weight0"],
                self.weights.get(f"{prefix}.sin") if kind == "up" else None)
        return self._edges[(index, kind)]

    def close(self):
        self._blocks.clear()
        self._edges.clear()
        # every name is set by `_upload_edges`, so nothing here is conditional
        for name in WEIGHT_NAMES:
            delattr(self, name)