"""
xmxrt - the device-resident runtime: buffers that persist, dispatches that batch.

`xmx.py` runs one GEMM per submit and hands the result back to the host, which is
why the XMX path does not beat a good CPU BLAS: a 720p frame moves 16.46 GB across
the boundary and 62x the kernel's own time goes into the trip
(`notes/phase11-what-is-left.md`, `notes/phase13-torch-and-blas.md`).

Here the operands are device buffers addressed by pointer, the elementwise passes
run on the GPU too, and a whole chain records into one command buffer with a single
fence at the end. Because the APU's memory is shared and HOST_CACHED, a buffer's
contents are also a numpy array - `Buffer.view()` - so feeding an input or reading
an output is an address, not a transfer.

    rt = Recorder()
    a = rt.buffer_from(activations)          # float32, on device
    rt.begin()
    elementwise.to_half(rt, a, a16, n)
    gemm.gemm(rt, a16, w16, out, M, N, K)
    rt.submit()

The pass encoders live in `gemm`, `attention` and `elementwise` as functions of the
recorder; this module owns what is genuinely per-recording: the library, the switch
value object, the recorded-pass count, allocation, and submit.
"""

from __future__ import annotations

import numpy as np

from . import native
from .config import FuseConfig
from .memory import Buffer, CommandGraph, host_view, host_write
from .ops import GRAPH


class Recorder:
    """Records a chain of GPU passes and submits it once."""

    def __init__(self, config=None, lib=None):
        self.lib = lib if lib is not None else native.load()
        self.recorded = 0
        self.config = config if config is not None else FuseConfig.from_environment(self.lib)

    # -- identity ------------------------------------------------------

    def with_config(self, **changes):
        """Replace switches for recordings made after this, without a new device.

        The configuration is frozen so a capture cannot replay under switches it was
        not made with; a caller that means to change one - a test comparing the fused
        and decomposed paths - says so here, and the next `graph_key()` reflects it.
        """
        self.config = self.config.with_config(**changes)
        return self

    def graph_key(self):
        return self.config.key(self.lib.xmx_specialization())

    def scratch_key(self):
        return self.config.scratch_key(self.lib.xmx_specialization())

    @property
    def buffer_bytes(self):
        """Live resident buffer sizes, excluding driver allocations and host weights."""
        return self.lib.xmx_buf_total_bytes()

    # -- allocation ----------------------------------------------------

    @property
    def staging(self):
        """Whether the graph's buffers are unmapped, so the host reaches them by copies."""
        return bool(self.lib.xmx_staging_mode())

    def buffer(self, count, dtype=np.float32, kind=GRAPH):
        return Buffer(int(count) * np.dtype(dtype).itemsize, kind=kind)

    # the same two helpers as methods, for callers outside this package: `src/ref` reaches
    # a runtime through an object, not through an import path
    def read(self, buffer, dtype=np.float32, shape=None, count=None):
        return host_view(buffer, dtype, shape, count)

    def write(self, buffer, array, rows=None):
        return host_write(buffer, array, rows)

    def buffer_from(self, array, dtype=np.float32, pad=0):
        """A device buffer holding `array`, optionally padded with zeros at the end."""
        array = np.ascontiguousarray(array, dtype=dtype)
        buffer = Buffer((array.size + int(pad)) * array.dtype.itemsize)
        if buffer.mapped:
            flat = buffer.view(dtype)
            flat[: array.size] = array.reshape(-1)
            if pad:
                flat[array.size :] = 0
        else:
            # the weights go up once per extent, not once per frame; a staged copy is the
            # right cost here and the only one available with no mapping
            if pad:
                buffer.zero()
            buffer.upload(array.reshape(-1))
        return buffer

    # -- recording -----------------------------------------------------

    def note(self, count=1):
        """One more pass in this recording. Pass encoders call this instead of counting."""
        self.recorded += count

    def begin(self):
        if self.lib.xmx_begin() != 0:
            raise RuntimeError("xmx_begin: " + self.lib.xmx_error().decode())
        self.recorded = 0
        return self

    def prepare_qkv(self, source, q, k, v, scale, windows, tokens, heads, *, vit=False):
        if min(windows, tokens, heads) <= 0:
            raise ValueError("QKV extents must be positive")
        if (
            len({q.id, k.id, v.id}) != 3
            or source.id in (q.id, k.id, v.id)
            or scale.id in (q.id, k.id, v.id)
        ):
            raise ValueError("QKV outputs must be separate from each other and the input")
        rows = windows * tokens * heads
        if source.nbytes < rows * 3 * 32 * 4 or scale.nbytes < heads * 4:
            raise ValueError("QKV input or scale buffer is too small")
        if any(buf.nbytes < rows * 32 * 2 for buf in (q, k, v)):
            raise ValueError("QKV output buffer is too small")
        if self.lib.xmx_rec_qkv(
            source.id, q.id, k.id, v.id, scale.id, rows, tokens, heads, int(vit)
        ):
            raise native.failure(self.lib, "xmx_rec_qkv")
        self.recorded += 1
        return self

    def independent(self):
        """A `with` block whose dispatches are known not to depend on one another.

        Everything inside records without a barrier between; leaving the block emits
        one. The graph has several such runs - a branched feed-forward's per-head GEMMs
        write disjoint slices of one buffer, the three head splits read one buffer and
        write three - and on small dispatches the overlap is worth having.
        """
        return _Independent(self)

    def capture(self):
        """Finish recording without executing; return reusable commands."""
        return CommandGraph(self.lib)

    def abort(self):
        """Discard unfinished recording after an error; completed graphs survive."""
        if self.lib.xmx_abort() != 0:
            raise RuntimeError("xmx_abort: " + self.lib.xmx_error().decode())

    def copy(self, source, target, nbytes, source_offset=0, target_offset=0):
        if min(nbytes, source_offset, target_offset) < 0:
            raise ValueError("copy sizes and offsets must be nonnegative")
        if self.lib.xmx_rec_copy(source.id, target.id, nbytes, source_offset, target_offset) != 0:
            raise RuntimeError("xmx_rec_copy: " + self.lib.xmx_error().decode())
        self.recorded += 1
        return self

    def specialize(self, mask=7):
        """Select cached shader variants: GEMM=1, elementwise=2, rows=4.

        The default enables all three. Zero retains the generic shaders for exact
        A/B tests on the same buffers. Switch only between command buffers.
        """
        if self.lib.xmx_specialize(mask) != 0:
            raise RuntimeError("xmx_specialize: " + self.lib.xmx_error().decode())
        return self

    def submit(self):
        count = self.lib.xmx_submit()
        if count < 0:
            raise RuntimeError("xmx_submit: " + self.lib.xmx_error().decode())
        return count


class _Independent:
    __slots__ = ("runtime",)

    def __init__(self, runtime):
        self.runtime = runtime

    def __enter__(self):
        self.runtime.lib.xmx_sync(0)
        return self.runtime

    def __exit__(self, *exc):
        self.runtime.lib.xmx_sync(1)
