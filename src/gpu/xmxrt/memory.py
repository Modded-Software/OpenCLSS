"""Device buffers and replayable command graphs.

A `Buffer`'s contents are a numpy array when the host can see the card's memory, and a
refusal when it cannot; `host_view`/`host_write` are the two operations that work either
way, by copying through staging when they must.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np

from . import native
from .ops import GRAPH


class Buffer:
    """A device buffer, addressable as a numpy array when the host can see it.

    `kind` says what it is for. `HOST_READ` and `HOST_WRITE` are the buffers the host
    touches - always mapped, cached for reading. `GRAPH` follows the device: mapped where
    the host can see the card's memory, device-local and **unmapped** where it cannot, and
    then `view()` refuses rather than handing back memory the device will not see.

    That refusal is the point. The alternative - a host shadow that someone must remember
    to upload - was tried by somebody else's port of this and produced frames rendered from
    data that never left the host, with no error anywhere. A buffer either is addressable or
    says so.
    """

    __slots__ = ("_lib", "id", "kind", "nbytes")

    def __init__(self, nbytes, kind=GRAPH):
        self._lib = native.load()
        self.id = self._lib.xmx_buf_create_kind(int(nbytes), int(kind))
        if self.id < 0:
            raise native.failure(self._lib, "xmx_buf_create")
        self.nbytes = int(nbytes)
        self.kind = int(kind)

    @property
    def mapped(self):
        return bool(self._lib.xmx_buf_host_visible(self.id))

    def view(self, dtype=np.float32, shape=None):
        pointer = self._lib.xmx_buf_ptr(self.id)
        if not pointer:
            raise RuntimeError(
                "this buffer is in device memory the host cannot address: use upload() or "
                "download(), or a HOST_READ/HOST_WRITE buffer the graph copies through"
            )
        count = self.nbytes // np.dtype(dtype).itemsize
        array = np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.nbytes,)
        )
        array = array.view(dtype)[:count]
        return array if shape is None else array[: int(np.prod(shape))].reshape(shape)

    def upload(self, array, offset=0):
        """Host data into this buffer: a memcpy when mapped, a staged copy when not."""
        array = np.ascontiguousarray(array)
        if (
            self._lib.xmx_buf_upload(self.id, array.ctypes.data, int(offset), int(array.nbytes))
            != 0
        ):
            raise native.failure(self._lib, "xmx_buf_upload")
        return self

    def download(self, dtype=np.float32, count=None, offset=0):
        out = np.empty(
            count if count is not None else self.nbytes // np.dtype(dtype).itemsize,
            dtype,
        )
        if self._lib.xmx_buf_download(self.id, out.ctypes.data, int(offset), int(out.nbytes)) != 0:
            raise native.failure(self._lib, "xmx_buf_download")
        return out

    def zero(self):
        # the library decides how: a memset when mapped, a recorded fill when not, because
        # the scratch arena clears roles in the middle of the recording that will use them
        if self._lib.xmx_buf_zero(self.id) != 0:
            raise native.failure(self._lib, "xmx_buf_zero")
        return self

    def free(self):
        if self.id >= 0:
            self._lib.xmx_buf_destroy(self.id)
            self.id = -1

    def __del__(self):
        try:
            self.free()
        except Exception as error:
            sys.stderr.write(f"xmxrt: free during teardown failed: {error}\n")


def host_view(buffer, dtype=np.float32, shape=None, count=None):
    """Read a buffer from the host, addressable or not.

    A mapped buffer hands back its own memory, as it always did. An unmapped one - the
    graph's buffers on a card whose memory the host cannot see - is copied out through
    staging. Tests and diagnostics read buffers the graph never sends back, and should not
    have to know which kind they are holding.
    """
    if buffer.mapped:
        array = buffer.view(dtype, shape)
        return array if count is None else array[:count]
    array = buffer.download(dtype, count)
    return array if shape is None else array[: int(np.prod(shape))].reshape(shape)


def host_write(buffer, array, rows=None):
    """Put host data into a buffer, addressable or not.

    `rows` is for the padded scratch layouts: the first rows of a `(padded, channels)`
    buffer are contiguous at its start, so writing them is one copy either way.
    """
    array = np.ascontiguousarray(array)
    if buffer.mapped:
        if rows is None:
            buffer.view(dtype=array.dtype, shape=array.shape)[...] = array
        else:
            buffer.view(dtype=array.dtype, shape=rows)[: array.shape[0]] = array
        return buffer
    return buffer.upload(array)


class CommandGraph:
    """Replayable commands. Referenced buffers must outlive this object."""

    def __init__(self, lib):
        self._lib = lib
        self.id = lib.xmx_graph_capture()
        if self.id < 0:
            raise native.failure(lib, "xmx_graph_capture")

    def run(self):
        passes = self._lib.xmx_graph_run(self.id)
        if passes < 0:
            raise native.failure(self._lib, "xmx_graph_run")
        return passes

    def free(self):
        if self.id >= 0:
            self._lib.xmx_graph_destroy(self.id)
            self.id = -1

    def __del__(self):
        try:
            self.free()
        except Exception as error:
            sys.stderr.write(f"xmxrt: free during teardown failed: {error}\n")
