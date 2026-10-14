#!/usr/bin/env python3
"""
nr_frame_resident - the whole 71-block graph on the device.

`nr_resident` records a block; this records a frame. Activations never come back to
the host: the stem writes a device buffer, every block, transition and skip reads and
writes device buffers, and only the head is read out.

Levels, for a network extent (H, W). The encoder halves five times and the decoder
mirrors it; the two deepest transitions pad to a multiple of the window first.

    L0  H     x W      C=32     block 0, and block 70 at the end
    L1  H/2   x W/2    C=32     blocks 1-3,   67-69
    L2  H/4   x W/4    C=64     blocks 5-7,   63-65
    L3  H/8   x W/8    C=128    blocks 9-13,  57-61
    L4  H/16  x W/16   C=256    blocks 15-21, 49-55
    L5  pad8(L4)/2     C=512    blocks 23-30, 40-47
    L6  pad8(L5)/2     C=1024   blocks 31-38, every token attending to every other

A block is safe writing over its own input - its target is touched only by the final
residual, after every read of the source - so a level needs one value buffer, not two.
"""

from __future__ import annotations

import os
import sys

import nr_resident as R
import numpy as np
from nr_frame_weights import WEIGHT_NAMES, DeviceWeights
from xmxrt import elementwise, gemm, memory, ops, scratch

try:
    import nr_image
except ImportError:  # pragma: no cover
    nr_image = None


def _native_half(source, target):
    """float32 into a half array natively; None when the library is not there."""
    return None if nr_image is None else nr_image.to_half(source, target)


host_copy = memory.host_view  # diagnostics read buffers the graph never returns

ENCODER = (
    (range(1, 4), 4, 1),
    (range(5, 8), 8, 2),
    (range(9, 14), 14, 4),
    (range(15, 22), 22, 8),
)
DECODER = (
    (48, range(49, 56), 4, 8),
    (56, range(57, 62), 3, 4),
    (62, range(63, 66), 2, 2),
    (66, range(67, 70), 1, 1),
)


def pad8(extent):
    return -(-extent // 8) * 8


class _HeadLease:
    """A block of host memory lent out as one head's buffer (`ResidentFrame.read_head`).

    Every array over it holds the buffer, views included, and NumPy releases it only when the
    last of them is gone. Then the block goes back to the frame for the next head: what the
    caller can still see is never written again, and nothing has to say when it is done.
    """

    def __init__(self, block, free):
        self.block, self.free = block, free

    def __buffer__(self, flags):
        return memoryview(self.block)

    def __release_buffer__(self, view):
        view.release()
        self.free.append(self.block)


_LEASES = sys.version_info >= (3, 12)  # PEP 688: a buffer exporter written in Python


class ResidentFrame:
    """The graph and its buffers, for one network extent. The weights are shared."""

    # the six named weight buffers live on `DeviceWeights` now; the body of a frame still
    # says `self.adapter`, because where they are kept is not that code's business
    WEIGHT_NAMES = WEIGHT_NAMES

    def __getattr__(self, name):
        if name in ResidentFrame.WEIGHT_NAMES:
            return getattr(self.w, name)
        raise AttributeError(name)

    def __init__(self, runtime, weights, height, width):
        self.split = (0.0, 0.0, 0.0)  # host write, graph, host read, of the last frame
        # a caller may hand over shared weights or the raw tensors; the benches and tests
        # hand over tensors, and then this frame owns the upload as it always did
        self.w = weights if isinstance(weights, DeviceWeights) else DeviceWeights(runtime, weights)
        self._owns_weights = self.w is not weights
        self.rt, self.weights = runtime, self.w.weights
        self.height, self.width = height, width
        self.levels = self._plan(height, width)
        self._scratch, self._blocks, self._buffers, self._edges = {}, {}, {}, {}
        self._graphs = {}
        self._arena = (
            scratch.ScratchArena(runtime)
            if os.environ.get("NR_SCRATCH_ARENA", "1") != "0"
            else None
        )
        self._planned = None  # what the arena was planned for: `_prepare_scratch`
        self._closed = False
        self._head_free = []  # host blocks a returned head was in and nobody holds now
        # Deep-level caching (notes/phase76): run the 16h-32h half on every Nth frame and
        # reuse its d5 on the frames between. 0 or 1 = off.
        self._deep_n = int(os.environ.get("NR_DEEP_CACHE", "0"))
        self._deep_tick = 0
        self._deep_ready = False

    @staticmethod
    def _plan(height, width):
        levels = [
            (height, width, 32),
            (height // 2, width // 2, 32),
            (height // 4, width // 4, 64),
            (height // 8, width // 8, 128),
            (height // 16, width // 16, 256),
        ]
        h, w = pad8(levels[4][0]) // 2, pad8(levels[4][1]) // 2
        levels.append((h, w, 512))
        levels.append((pad8(h) // 2, pad8(w) // 2, 1024))
        return levels

    # -- lazily built and reused -----------------------------------------

    def block(self, index, heads, family="window"):
        return self.w.block(index, heads, family)

    def scratch(self, block, height, width, tokens=None):
        key = (
            height,
            width,
            tokens,
            block.channels,
            block.heads,
            getattr(block, "split", False),
            getattr(block, "branched", False),
        )
        if key not in self._scratch:
            self._scratch[key] = (
                R.GlobalScratch(self.rt, block, tokens, arena=self._arena)
                if tokens
                else R.BlockScratch(self.rt, block, height, width, arena=self._arena)
            )
        return self._scratch[key]

    def transition_scratch(self, elements):
        rounded = 1 << max(1, int(elements - 1)).bit_length()
        if rounded not in self._edges:
            self._edges[rounded] = R.TransitionScratch(self.rt, rounded, 0, arena=self._arena)
        return self._edges[rounded]

    # Two buffers in the whole graph are touched by the host, once each per frame: the
    # features go in and the head comes back. They are named here so they can be put where
    # the host can reach them - cached for the strided read of the head - while everything
    # else follows the device, which on a discrete card means the card's own memory.
    HOST_SIDE = {
        "features": ops.HOST_WRITE,
        "features_host16": ops.HOST_WRITE,
        "head": ops.HOST_READ,
        "head4": ops.HOST_READ,
    }

    def input_buffer(self):
        if self.rt.config.input_fp16:
            return self.buffer("features_host16", self.height * self.width * 16, np.float16)
        return self.buffer("features", self.height * self.width * 16)

    def input_view(self):
        """The mapped input itself, (height, width, 16): features built here need no copy.
        The buffer is HOST_WRITE, which is mapped in every memory mode - a downloaded copy
        would take the features and drop them, so an unmapped one is refused."""
        buffer = self.input_buffer()
        if not getattr(buffer, "mapped", True):
            raise RuntimeError("the input buffer is not mapped")
        return buffer.view(
            np.float16 if self.rt.config.input_fp16 else np.float32,
            (self.height, self.width, 16),
        )

    def head_buffer(self):
        return self.buffer(
            "head4" if self.rt.config.compact_head else "head",
            self.height * self.width * (4 if self.rt.config.compact_head else 16),
        )

    def read_head(self):
        channels = 4 if self.rt.config.compact_head else 16
        view = memory.host_view(self.head_buffer(), shape=(self.height, self.width, channels))[
            ..., :4
        ]
        if not _LEASES:
            return np.array(view, copy=True)
        # A fresh array a frame costs a page fault every 4 KB on Windows, whose heap gives
        # memory this size back to the system when it is freed: 15 ms of a 720p frame, of
        # which the copy is 2.4. So a head is lent from a block the frame keeps, and a block
        # goes out again only once every array and view over the last head in it is gone.
        block = self._head_free.pop() if self._head_free else np.empty(view.size, view.dtype)
        head = np.frombuffer(_HeadLease(block, self._head_free), view.dtype).reshape(view.shape)
        np.copyto(head, view)
        return head

    def buffer(self, name, elements, dtype=np.float32):
        existing = self._buffers.get(name)
        if existing is None or existing.nbytes < elements * np.dtype(dtype).itemsize:
            if existing is not None:
                existing.free()
            existing = self.rt.buffer(elements, dtype, kind=self.HOST_SIDE.get(name, ops.GRAPH))
            self._buffers[name] = existing
        return existing

    def edge(self, index, kind):
        return self.w.edge(index, kind)

    def _prepare_scratch(self, capture=None):
        """Plan every role's size before a buffer address can be recorded.

        Every block's scratch plans every buffer it might use; a recording made once with
        stand-in addresses (`ScratchArena.discover`) finds the ones it does use, and only
        those are allocated. Which ones depends on the switches in force and on whether a
        capture is wanted - a capture keeps block 0's stem - so either changing plans the
        arena again, dropping the graphs recorded against the old one."""
        if self._arena is None:
            return
        plan = (self.rt.scratch_key(), capture is not None)
        if self._arena.sealed:
            if plan == self._planned:
                return
            for graph in self._graphs.values():
                graph.free()
            self._graphs.clear()
            self._scratch.clear()
            self._edges.clear()
            self._arena.free()
            self._arena = scratch.ScratchArena(self.rt)
        for index in (0, 70):
            self.scratch(self.block(index, 1), self.height, self.width)
        for level, (regular, transition, heads) in enumerate(ENCODER, 1):
            h, w, channels = self.levels[level]
            for index in (*regular, transition):
                self.scratch(self.block(index, heads), h, w)
            self.transition_scratch(pad8(h) * pad8(w) * channels)
        h, w, channels = self.levels[5]
        for index in (*range(23, 31), *range(40, 48)):
            self.scratch(self.block(index, 16, "split"), h, w)
        self.transition_scratch(pad8(h) * pad8(w) * channels)
        self.transition_scratch(h * w * channels)
        gh, gw, _ = self.levels[6]
        for index in range(31, 39):
            self.scratch(self.block(index, 32, "global"), gh, gw, tokens=gh * gw)
        for transition, regular, level, heads in DECODER:
            sh, sw, schannels = self.levels[level]
            self.transition_scratch(sh * sw * max(channels, schannels))
            for index in (transition, *regular):
                self.scratch(self.block(index, heads), sh, sw)
            channels = schannels
        with self._arena.discover():
            self._run(None, None, capture, None, None, dry=True)
        self._arena.seal()
        self._planned = plan

    def close(self):
        """Release recorded commands before any buffers they reference."""
        for graph in self._graphs.values():
            graph.free()
        self._graphs.clear()
        self._closed = True
        self._head_free.clear()
        for cache in (self._scratch, self._blocks, self._buffers, self._edges):
            cache.clear()
        if self._arena is not None:
            self._arena.free()
        # shared weights outlive the frame; ones this frame uploaded itself do not
        if self._owns_weights:
            self.w.close()

    def __del__(self):
        try:
            self.close()
        except Exception as error:
            sys.stderr.write(f"nr_frame_resident: close during teardown failed: {error}\n")

    # -- the frame --------------------------------------------------------

    def run(self, features, submits=None, capture=None, timing=None, execution=None):
        try:
            return self._run(features, submits, capture, timing, execution)
        except Exception:
            self.rt.abort()
            raise

    def _run(self, features, submits, capture, timing, execution, dry=False):
        rt = self.rt
        import time as _time

        if self._closed:
            raise RuntimeError("ResidentFrame is closed")
        if dry:
            # the scratch plan's discovery: recorded as a replay would be, never run
            execution = "replay"
        else:
            if features.shape != (self.height, self.width, 16):
                raise ValueError("features must match the frame's (height, width, 16)")
            self._prepare_scratch(capture)
            execution = execution or os.environ.get("NR_FRAME_MODE", "replay")
        if execution not in ("block", "single", "replay"):
            raise ValueError("NR_FRAME_MODE must be block, single or replay")
        # Diagnostic reads require a fence after each block.
        if not dry and (capture is not None or timing is not None):
            execution = "block"
        batched = execution != "block"
        staged = rt.staging
        # Deep-level caching: the 16h-32h half runs only on a refresh frame, and its d5
        # is reused on the frames between. A dry run always takes the full path, so the
        # scratch plan still sees every role.
        deep = self._deep_n
        refresh = dry or deep <= 1 or not self._deep_ready or self._deep_tick % deep == 0
        key = (rt.graph_key(), refresh)
        # Counted here, not at the end: the replay fast path below returns before the end,
        # and a frozen counter would pin every later frame to one cache phase.
        if not dry:
            self._deep_tick += 1
        recording = False

        def begin():
            nonlocal recording
            if not batched or not recording:
                rt.begin()
                recording = True

        def keep(name, buffer, count, shape=None, dtype=np.float32):
            if capture is not None and not dry:
                data = host_copy(buffer, dtype, count).astype(np.float32)
                capture[name] = data if shape is None else data.reshape(shape)

        height, width = self.height, self.width
        pixels = height * width
        counter = [0]

        stage = ["stem"]

        def submit():
            if batched:
                return
            started = _time.perf_counter()
            counter[0] += rt.submit()
            if timing is not None:
                timing.setdefault(stage[0], [0.0, 0])
                timing[stage[0]][0] += _time.perf_counter() - started
                timing[stage[0]][1] += 1

        # Separate names keep captured graphs' addresses valid when switching modes.
        # HOST_WRITE is always mapped, including when graph buffers use staging.
        source = self.input_buffer()
        # Wall times around host writes, graph completion and host reads. They are
        # not PCIe counters: GPU access to mapped host memory occurs during the graph.
        mark = _time.perf_counter()
        view = None if dry else self.input_view()
        if dry:
            pass  # nothing runs, so nothing goes in
        elif features.ctypes.data == view.ctypes.data and features.dtype == view.dtype:
            pass  # built in place (`input_view`): nothing to copy
        elif rt.config.input_fp16:
            # Convert directly into the mapped input: no temporary half array and no
            # FP32 buffer crosses the host/device boundary before a GPU to_half pass.
            # Natively where it can - NumPy's cast was slower than the pass it saved.
            flat = np.ascontiguousarray(features, dtype=np.float32).reshape(pixels, 16)
            if _native_half(flat, view.reshape(pixels, 16)) is None:
                np.copyto(view.reshape(pixels, 16), flat, casting="unsafe")
        else:
            memory.host_write(source, features.reshape(-1, 16), rows=(pixels, 16))
        carried = _time.perf_counter() - mark
        if execution == "replay" and key in self._graphs and not dry:
            mark = _time.perf_counter()
            passes = self._graphs[key].run()
            ran = _time.perf_counter() - mark
            if submits is not None:
                submits.append(passes)
            mark = _time.perf_counter()
            out = self.read_head()
            self.split = (carried, ran, _time.perf_counter() - mark)
            return out.reshape(height, width, 4)
        begin()
        if rt.config.input_fp16:
            source16 = source
        else:
            source16 = self.buffer("features16", pixels * 16, np.float16)
            elementwise.to_half(rt, source, source16, pixels * 16)
        block0 = self.block(0, 1)
        scratch0 = self.scratch(block0, height, width)
        # The stem is read only by block 0's feed-forward, which can make it itself - then
        # neither width of it is ever stored (`ffn_fused_stem`). A capture keeps the
        # stored stem, since it wants to look at it.
        made_stem = (
            rt.config.fuse_glue and capture is None and R.can_make_stem(rt, block0, scratch0)
        )
        stem = None if made_stem else self.buffer("stem", pixels * 32)
        if made_stem:
            pass
        elif rt.config.fuse_glue:
            # the stem as block 0's residual needs it and as its first GEMM reads it
            gemm.gemm_dual(rt, source16, self.adapter, stem, scratch0.value16, pixels, 32, 16)
        else:
            gemm.gemm(rt, source16, self.adapter, stem, pixels, 32, 16)
        submit()

        # block 0 runs at full resolution; its output is both the skip the post block
        # merges and, pooled, the encoder's input
        stage[0] = "block0 + pool"
        # Block 0's output has two readers, the pool into the encoder and the published
        # skip the last block merges, and its closing residual can serve both itself
        # (`gemm_residual_pool`); then the float32 output is never stored. A capture keeps
        # it, since it wants to look at it.
        pooled_here = (
            rt.config.fuse_glue and capture is None and R.can_pool_output(rt, block0, scratch0)
        )
        raw = None if pooled_here else self.buffer("block0", pixels * 32)
        # Everything the graph publishes is E4M3, which is exact in float16, so every
        # published buffer is stored narrow: half the traffic, and the widening pass in
        # front of each block's first GEMM disappears. `src/bench/bf16_check.py`.
        full_skip = self.buffer("full_skip", pixels * 32, np.float16)
        h, w, channels = self.levels[1]
        value = self.buffer("l1", h * w * 32, np.float16)
        begin()
        R.record_block(
            rt,
            block0,
            scratch0,
            source=stem,
            target=raw,
            source16=scratch0.value16 if rt.config.fuse_glue and not made_stem else None,
            stem=(source16, self.adapter) if made_stem else None,
            pool=(value, full_skip) if pooled_here else None,
        )
        # the post block's skip is block 0 published; the encoder pools the
        # *unpublished* output, so both come from `raw` and neither from the other
        if pooled_here:
            pass
        elif rt.config.fuse_glue and height % 2 == 0 and width % 2 == 0:
            elementwise.pool2_skip(rt, raw, value, full_skip, height, width, 32)
        else:
            with rt.independent():
                elementwise.e4m3_half(rt, raw, full_skip, pixels * 32)
                elementwise.pool2(
                    rt,
                    raw,
                    value,
                    height,
                    width,
                    32,
                    epilogue=ops.EPI_E4M3,
                    narrow=True,
                )
        submit()
        if stem is not None:
            keep("stem", stem, pixels * 32, (1, height, width, 32))
        if raw is not None:
            keep("block0", raw, pixels * 32, (1, height, width, 32))
        keep("full_skip", full_skip, pixels * 32, (1, height, width, 32), np.float16)
        keep("l1_in", value, h * w * 32, (1, h, w, 32), np.float16)

        skips, level = {}, 1
        for regular, transition, heads in ENCODER:
            h, w, channels = self.levels[level]
            for index in regular:
                stage[0] = f"encoder L{level} blocks (C={channels})"
                block = self.block(index, heads)
                begin()
                R.record_block(
                    rt,
                    block,
                    self.scratch(block, h, w),
                    source=value,
                    target=value,
                    publish=ops.EPI_E4M3,
                    source_half=True,
                    target_half=True,
                )
                submit()
            keep(f"l{level}", value, h * w * channels, (1, h, w, channels), np.float16)

            block = self.block(transition, heads)
            edge = self.edge(transition, "down")
            nh, nw, nchannels = self.levels[level + 1]
            unpublished = self.buffer("unpublished", h * w * channels)
            nxt = self.buffer(f"l{level + 1}", nh * nw * nchannels, np.float16)
            padded = pad8(h) * pad8(w) * channels
            stage[0] = f"downsample L{level}->L{level + 1}"
            begin()
            R.record_block(
                rt,
                block,
                self.scratch(block, h, w),
                source=value,
                target=unpublished,
                source_half=True,
            )
            # The level's skip is the transition block's own output, published - not the
            # block before it, which is what MLX-DLSS's graph merged (notes/opendlss-
            # reference.md). It goes into the level's buffer, which nothing reads again
            # once the transition block has: the decoder writes d1-d4, never l1-l4.
            elementwise.e4m3_half(rt, unpublished, value, h * w * channels)
            skips[level] = value
            R.record_downsample(
                rt,
                edge,
                self.transition_scratch(padded),
                unpublished,
                nxt,
                h,
                w,
                channels,
                pad_to=8 if transition == 22 else 0,
                target_half=True,
            )
            submit()
            value, level = nxt, level + 1
            keep(
                f"l{level}_in",
                value,
                nh * nw * nchannels,
                (1, nh, nw, nchannels),
                np.float16,
            )

        # the split family, then the bottleneck
        h, w, channels = self.levels[5]
        # Deep-level caching (notes/phase76): everything from here to the end of blocks
        # 40-47 is the 16h-32h half - the frame's most expensive part and the part of a
        # temporal model that changes slowest. On a non-refresh frame, reuse the d5 the
        # last full pass left, and record none of it.
        deep_cache = self.buffer("d5_cache", h * w * channels, np.float16) if deep > 1 else None
        if not refresh:
            value = self.buffer("d5", h * w * channels, np.float16)
            begin()
            rt.copy(deep_cache, value, h * w * channels * 2)
            submit()
        else:
            stage[0] = "split blocks 23-30 (C=512)"
            # Block 30's output has two readers: the decoder's skip takes it published, the
            # bottleneck's pool raw (notes/opendlss-reference.md).
            raw30 = self.buffer("raw30", h * w * channels)
            for index in range(23, 31):
                block = self.block(index, 16, "split")
                last = index == 30
                begin()
                R.record_block(
                    rt,
                    block,
                    self.scratch(block, h, w),
                    source=value,
                    target=raw30 if last else value,
                    publish=0 if last else ops.EPI_E4M3,
                    source_half=True,
                    target_half=not last,
                )
                if last:
                    elementwise.e4m3_half(rt, raw30, value, h * w * channels)
                submit()
            keep("l5", value, h * w * channels, (1, h, w, channels), np.float16)
            # l5 itself is the skip, as for the levels above: the decoder input merge below
            # writes d5 rather than l5, which is what the copy was protecting.
            split_skip = value

            gh, gw, gchannels = self.levels[6]
            tokens = gh * gw
            # With the glue fused the eight blocks run in their scratch's half value itself
            # (`record_global_block`, `chain`), and the downsample writes it.
            chain = rt.config.fuse_glue
            deep_buf = (
                self.scratch(self.block(31, 32, "global"), gh, gw, tokens=tokens).io16
                if chain
                else self.buffer("l6", gh * gw * gchannels, np.float16)
            )
            begin()
            # pooled raw and published before its GEMM, as the encoder's transitions are
            R.record_downsample(
                rt,
                self.bottleneck,
                self.transition_scratch(pad8(h) * pad8(w) * channels),
                raw30,
                deep_buf,
                h,
                w,
                channels,
                pad_to=8,
                target_half=True,
            )
            submit()

            stage[0] = "global blocks 31-38 (C=1024)"
            for index in range(31, 39):
                block = self.block(index, 32, "global")
                scratch = self.scratch(block, gh, gw, tokens=tokens)
                begin()
                if chain:
                    R.record_global_block(rt, block, scratch, chain=True)
                    submit()
                    continue
                # Published values are exact in both widths. Convert on-device when
                # batching; the block mode retains the original host-copy reference.
                if batched or staged:
                    elementwise.from_half(rt, deep_buf, scratch.value, tokens * gchannels)
                else:
                    scratch.value.view()[: tokens * gchannels] = deep_buf.view(np.float16)[
                        : tokens * gchannels
                    ]
                R.record_global_block(rt, block, scratch)
                elementwise.e4m3(rt, scratch.out, scratch.out, scratch.padded * gchannels)
                if batched or staged:
                    elementwise.to_half(rt, scratch.out, deep_buf, tokens * gchannels)
                submit()
                if not batched and not staged:
                    deep_buf.view(np.float16)[: tokens * gchannels] = scratch.out.view()[
                        : tokens * gchannels
                    ]

            keep("l6", deep_buf, tokens * gchannels, (1, gh, gw, gchannels), np.float16)

            # the decoder input merge, then the split family again
            value = self.buffer("d5", h * w * channels, np.float16)
            begin()
            R.record_upsample_merge(
                rt,
                self.decoder_input,
                self.transition_scratch(h * w * channels),
                deep_buf,
                split_skip,
                value,
                gh,
                gw,
                h,
                w,
                gchannels,
                channels,
                source_half=True,
                skip_half=True,
                target_half=True,
            )
            submit()
            stage[0] = "split blocks 40-47 (C=512)"
            for index in range(40, 48):
                block = self.block(index, 16, "split")
                begin()
                R.record_block(
                    rt,
                    block,
                    self.scratch(block, h, w),
                    source=value,
                    target=value,
                    publish=ops.EPI_E4M3,
                    source_half=True,
                    target_half=True,
                )
                submit()
            keep("d5", value, h * w * channels, (1, h, w, channels), np.float16)
            if deep_cache is not None:
                begin()
                rt.copy(value, deep_cache, h * w * channels * 2)
                submit()
                self._deep_ready = True

        for transition, regular, skip_level, heads in DECODER:
            stage[0] = f"decoder upsample -> L{skip_level}"
            sh, sw, schannels = self.levels[skip_level]
            edge = self.edge(transition, "up")
            target = self.buffer(f"d{skip_level}", sh * sw * schannels, np.float16)
            block = self.block(transition, heads)
            # The 32-channel block takes its merge raw - its feed-forward's GEMM reads the
            # publication, its residual the raw value - where the branched ones take it
            # published as both (notes/opendlss-reference.md). The raw merge goes into the
            # level-1 transition's buffer, its size and long dead by now.
            raw = None if block.branched else self.buffer("unpublished", sh * sw * schannels)
            begin()
            R.record_upsample_merge(
                rt,
                edge,
                self.transition_scratch(sh * sw * max(channels, schannels)),
                value,
                skips[skip_level],
                target if raw is None else raw,
                h,
                w,
                sh,
                sw,
                channels,
                schannels,
                source_half=True,
                skip_half=True,
                target_half=raw is None,
                published=None if raw is None else target,
            )
            R.record_block(
                rt,
                block,
                self.scratch(block, sh, sw),
                source=target if raw is None else raw,
                source16=None if raw is None else target,
                target=target,
                publish=ops.EPI_E4M3,
                source_half=raw is None,
                target_half=True,
            )
            submit()
            value, h, w, channels = target, sh, sw, schannels
            for index in regular:
                stage[0] = f"decoder L{skip_level} blocks (C={channels})"
                block = self.block(index, heads)
                begin()
                R.record_block(
                    rt,
                    block,
                    self.scratch(block, h, w),
                    source=value,
                    target=value,
                    publish=ops.EPI_E4M3,
                    source_half=True,
                    target_half=True,
                )
                submit()
            keep(
                f"d{skip_level}",
                value,
                h * w * channels,
                (1, h, w, channels),
                np.float16,
            )

        # back to full resolution, merged with block 0's output, then the head
        stage[0] = "block70 + head"
        block70 = self.block(70, 1)
        scratch70 = self.scratch(block70, height, width)
        # The head reads block 70's output as half, and nothing reads it as float32, so
        # the block's closing residual stores half itself: the same rounding the separate
        # to_half pass applied, without 126 MB of float32 written at 720p to be read once.
        # And with the fused window block the head itself comes out of block 70's own pass,
        # so neither the output nor its half copy is ever stored (window_block.comp).
        fused_head = (
            rt.config.fuse_head
            and capture is None
            and R.can_fuse_window_block(rt, block70, scratch70)
        )
        out16 = None if fused_head else self.buffer("out16", pixels * 32, np.float16)
        # Block 70's input is read only by its feed-forward, which can make it itself:
        # then neither width of it is ever stored (`ffn_fused_merge`).
        merge = None
        if rt.config.fuse_glue and R.can_merge_input(rt, block70, scratch70):
            merge = (value, full_skip, self.merge_sincos, w)
        merged = None if merge else self.buffer("merged", pixels * 32)
        begin()
        if merge:
            pass
        elif rt.config.fuse_glue:
            elementwise.upsample_merge(
                rt,
                value,
                full_skip,
                self.merge_sincos,
                merged,
                scratch70.value16,
                height,
                width,
                w,
                32,
            )
        else:
            upsampled = self.buffer("upsampled", pixels * 32)
            elementwise.upsample2(rt, value, upsampled, w, height, width, 32, a_half=True)
            elementwise.scale_channel(rt, upsampled, self.merge_sin, merged, pixels * 32, 32)
            elementwise.residual(
                rt,
                merged,
                full_skip,
                self.merge_cos,
                merged,
                pixels * 32,
                32,
                b_half=True,
            )
        R.record_block(
            rt,
            block70,
            scratch70,
            source=merged,
            target=out16,
            target_half=True,
            source16=scratch70.value16 if rt.config.fuse_glue and not merge else None,
            merge=merge,
            head=(self.head, self.head_buffer(), 4 if rt.config.compact_head else 16)
            if fused_head
            else None,
        )
        if not fused_head:
            gemm.gemm(
                rt,
                out16,
                self.head,
                self.head_buffer(),
                pixels,
                16,
                32,
                compact_output=rt.config.compact_head,
            )
        submit()

        if dry:
            rt.abort()
            return None
        if execution == "replay":
            self._graphs[key] = rt.capture()
            counter[0] = self._graphs[key].run()
        elif execution == "single":
            counter[0] = rt.submit()
        if submits is not None:
            submits.append(counter[0])
        return self.read_head()
