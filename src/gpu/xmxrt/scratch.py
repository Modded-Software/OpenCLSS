"""Scratch reuse: plan shared buffers by role, freeze sizes before recording."""

from __future__ import annotations

import contextlib

import numpy as np


class ScratchArena:
    """Plan shared scratch by role, then freeze sizes before commands are recorded.

    Blocks run sequentially and may share each role's storage. Input, output and
    encoder skips belong to the frame separately. Unused roles allocate no memory.

    Every block plans every buffer its recorders might use, and which ones they do
    use depends on the switches in force: the fused paths never touch the float32
    scores, the half probabilities or the float32 QKV projection, and those are the
    largest buffers in the frame. `discover()` records once with a stand-in address
    for each role and sizes the role by the buffers that recording touched - 2.2 GiB
    of scratch at 1920x1088 became under half a gigabyte.
    """

    # Within a block these roles have disjoint live intervals. Barriers already
    # separate their producers/last consumers in nr_resident. Transition scratch
    # runs between blocks and reuses those same allocations. Keep FFN residuals
    # separate: they stay live until the closing attention residual.
    ALIASES = {
        # key16 is K when the projection's own epilogue writes it: k16's role is the
        # projection's input, which other workgroups are still reading at that moment
        **dict.fromkeys(
            (
                "hidden16",
                "proj",
                "key16",
                "attended",
                "attention",
                "transition.padded",
                "transition.projected",
            ),
            "projection",
        ),
        **dict.fromkeys(("branch", "v16"), "branch_value"),
        **dict.fromkeys(
            (
                "value16",
                "win16",
                "ffn16",
                "k16",
                "transition.pooled16",
                "transition.projected16",
            ),
            "input_key",
        ),
        **dict.fromkeys(("heads16", "core16", "q16", "probs16", "merged16"), "query_probability"),
        **dict.fromkeys(
            ("scores", "context", "context16", "transition.upsampled"), "scores_context"
        ),
        **dict.fromkeys(("ffn", "transition.scaled"), "residual"),
    }

    def __init__(self, runtime):
        self.runtime = runtime
        self.buffers = {}
        self.sealed = False
        self._plan_state = [False]
        # while discovering, each touched role's stand-in buffer, by role
        self._stand_ins = [None]

    def buffer(self, name, count, dtype=np.float32):
        dtype = np.dtype(dtype)
        key = (self.ALIASES[name], "bytes") if name in self.ALIASES else (name, dtype.str)
        size = int(count) * dtype.itemsize
        if key not in self.buffers:
            if self.sealed:
                raise RuntimeError(f"scratch role was not planned: {name}")
            self.buffers[key] = _ScratchBuffer(self, size)
        buffer = self.buffers[key]
        if size > buffer.nbytes:
            if self.sealed:
                raise RuntimeError(f"scratch role exceeds its plan: {name}")
            buffer.nbytes = size
        return _ScratchHandle(buffer, key, name, size, self._stand_ins)

    @contextlib.contextmanager
    def discover(self):
        """Inside, a planned buffer's `id` is its role's stand-in and marks it used; on
        leaving, every role shrinks to the largest buffer used in it - nothing, if none
        was. A recording made inside must never run: its addresses are the stand-ins'."""
        if self.sealed or self._stand_ins[0] is not None:
            raise RuntimeError("scratch discovery comes once, before the seal")
        self._stand_ins[0] = {}
        try:
            yield self
        finally:
            for stand_in in self._stand_ins[0].values():
                stand_in.free()
            self._stand_ins[0] = None
        for buffer in self.buffers.values():
            buffer.nbytes = max((size for size, used in buffer.planned if used[0]), default=0)

    def seal(self):
        self.sealed = True
        self._plan_state[0] = True

    def free(self):
        for buffer in self.buffers.values():
            buffer.free()
        self.buffers.clear()


class _ScratchBuffer:
    def __init__(self, arena, nbytes):
        # Keep no reference back to the arena, so dropping a frame releases it.
        self.runtime = arena.runtime
        self._plan_state = arena._plan_state
        self.nbytes = nbytes
        self.planned = []  # (size, [used]) for every buffer planned in this role
        self._buffer = None
        self._zero = False
        self._closed = False

    def _get(self):
        if self._closed:
            raise RuntimeError("scratch buffer is closed")
        if not self._plan_state[0]:
            raise RuntimeError("scratch plan must be sealed before allocation")
        if self._buffer is None:
            self._buffer = self.runtime.buffer(self.nbytes, np.uint8)
            if self._zero:
                self._buffer.zero()
        return self._buffer

    @property
    def id(self):
        return self._get().id

    @property
    def mapped(self):
        return self._get().mapped

    def view(self, dtype=np.float32, shape=None):
        return self._get().view(dtype, shape)

    def upload(self, array, offset=0):
        return self._get().upload(array, offset)

    def download(self, dtype=np.float32, count=None, offset=0):
        return self._get().download(dtype, count, offset)

    def zero(self):
        self._zero = True
        if self._buffer is not None:
            self._buffer.zero()
        return self

    def free(self):
        if self._buffer is not None:
            self._buffer.free()
        self._closed = True


class _ScratchHandle:
    """One planned buffer: its own name and size, its role's storage.

    References only the role and a shared cell, never the arena or its siblings, so
    nothing here forms a cycle and a dropped frame releases its memory at once."""

    __slots__ = ("_key", "_role", "_stand_ins", "_used", "name", "nbytes")

    def __init__(self, role, key, name, nbytes, stand_ins):
        self._role, self._key, self.name, self.nbytes = role, key, name, nbytes
        self._used = [False]
        self._stand_ins = stand_ins
        role.planned.append((nbytes, self._used))

    def _get(self):
        if self._stand_ins[0] is not None:
            raise RuntimeError(f"scratch {self.name} reached from the host while discovering")
        if self.nbytes > self._role.nbytes:
            raise RuntimeError(f"scratch role exceeds its plan: {self.name}")
        return self._role._get()

    @property
    def id(self):
        stand_ins = self._stand_ins[0]
        if stand_ins is None:
            return self._get().id
        self._used[0] = True
        if self._key not in stand_ins:
            stand_ins[self._key] = self._role.runtime.buffer(256, np.uint8)
        return stand_ins[self._key].id

    @property
    def mapped(self):
        return self._get().mapped

    def view(self, dtype=np.float32, shape=None):
        return self._get().view(dtype, shape)

    def upload(self, array, offset=0):
        return self._get().upload(array, offset)

    def download(self, dtype=np.float32, count=None, offset=0):
        return self._get().download(dtype, count, offset)

    def zero(self):
        self._role.zero()
        return self

    def free(self):
        self._role.free()
