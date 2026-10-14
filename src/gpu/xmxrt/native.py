"""The native runtime library: one shared instance of `xmx_*`, its signatures bound.

Loading the library is a process-wide act - `xmx.py` loads it, this shares the
instance, device and queue - and every module that talks to the device goes through
`load()`. The two shader paths below (`XMX_*_SPV` overrides, the portable twins) are
the only place the `.spv` names appear, which is why the release tooling scans this
file for them.
"""

from __future__ import annotations

import ctypes
import os

import nr_build

_lib = None


class DeviceLost(RuntimeError):
    """`VK_ERROR_DEVICE_LOST`: the device went away under this process, normally the
    driver resetting a hung GPU.

    Unlike every other failure here it is final. Each later submit on this device fails
    the same way, so a caller should stop rather than retry; a new process gets a new
    device.
    """


def failure(lib, call):
    """The exception for a call that returned an error, told apart by the library's own
    record of a lost device rather than by the text of the message."""
    message = f"{call}: {lib.xmx_error().decode()}"
    return DeviceLost(message) if lib.xmx_device_lost() else RuntimeError(message)


def load():
    """The bound library, loaded once. Every caller shares it."""
    global _lib
    if _lib is not None:
        return _lib
    import xmx

    lib = xmx._load()  # shares the instance, device and queue
    lib.xmx_res_init.argtypes = [ctypes.c_char_p] * 6
    lib.xmx_res_init.restype = ctypes.c_int
    lib.xmx_buf_create.argtypes = [ctypes.c_ulonglong]
    lib.xmx_buf_create.restype = ctypes.c_int
    lib.xmx_buf_create_kind.argtypes = [ctypes.c_ulonglong, ctypes.c_int]
    lib.xmx_buf_create_kind.restype = ctypes.c_int
    lib.xmx_buf_host_visible.argtypes = [ctypes.c_int]
    lib.xmx_buf_host_visible.restype = ctypes.c_int
    lib.xmx_buf_zero.argtypes = [ctypes.c_int]
    lib.xmx_buf_zero.restype = ctypes.c_int
    lib.xmx_staging_mode.argtypes = []
    lib.xmx_staging_mode.restype = ctypes.c_int
    lib.xmx_buf_upload.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
    ]
    lib.xmx_buf_upload.restype = ctypes.c_int
    lib.xmx_buf_download.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
    ]
    lib.xmx_buf_download.restype = ctypes.c_int
    lib.xmx_buf_destroy.argtypes = [ctypes.c_int]
    lib.xmx_buf_destroy.restype = ctypes.c_int
    lib.xmx_begin.argtypes = []
    lib.xmx_begin.restype = ctypes.c_int
    lib.xmx_abort.argtypes = []
    lib.xmx_abort.restype = ctypes.c_int
    lib.xmx_sync.argtypes = [ctypes.c_int]
    lib.xmx_sync.restype = ctypes.c_int
    lib.xmx_specialize.argtypes = [ctypes.c_uint]
    lib.xmx_specialize.restype = ctypes.c_int
    lib.xmx_staged_partial.argtypes = [ctypes.c_uint]
    lib.xmx_staged_partial.restype = ctypes.c_int
    lib.xmx_staged32.argtypes = [ctypes.c_uint]
    lib.xmx_staged32.restype = ctypes.c_int
    lib.xmx_staged32_calls.argtypes = []
    lib.xmx_staged32_calls.restype = ctypes.c_int
    lib.xmx_staged_packed.argtypes = []
    lib.xmx_staged_packed.restype = ctypes.c_int
    lib.xmx_portable_packed.argtypes = []
    lib.xmx_portable_packed.restype = ctypes.c_int
    lib.xmx_staged32_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.xmx_staged32_init.restype = ctypes.c_int
    lib.xmx_rows_init.argtypes = [ctypes.c_char_p]
    lib.xmx_rows_init.restype = ctypes.c_int
    lib.xmx_specialized_count.argtypes = []
    lib.xmx_specialized_count.restype = ctypes.c_int
    lib.xmx_specialization.argtypes = []
    lib.xmx_specialization.restype = ctypes.c_int
    lib.xmx_graph_capture.argtypes = []
    lib.xmx_graph_capture.restype = ctypes.c_int
    lib.xmx_graph_run.argtypes = [ctypes.c_int]
    lib.xmx_graph_run.restype = ctypes.c_int
    lib.xmx_graph_destroy.argtypes = [ctypes.c_int]
    lib.xmx_graph_destroy.restype = ctypes.c_int
    lib.xmx_rec_copy.argtypes = [ctypes.c_int] * 2 + [ctypes.c_ulonglong] * 3
    lib.xmx_rec_copy.restype = ctypes.c_int
    lib.xmx_submit.argtypes = []
    lib.xmx_submit.restype = ctypes.c_int
    lib.xmx_rec_gemm.argtypes = [ctypes.c_int] * 3 + [ctypes.c_uint] * 14
    lib.xmx_rec_gemm.restype = ctypes.c_int
    lib.xmx_rec_gemm_residual.argtypes = [ctypes.c_int] * 5 + [ctypes.c_uint] * 4
    lib.xmx_rec_gemm_residual.restype = ctypes.c_int
    lib.xmx_rec_gemm_window_residual.argtypes = [ctypes.c_int] * 5 + [ctypes.c_uint] * 8
    lib.xmx_rec_gemm_window_residual.restype = ctypes.c_int
    lib.xmx_window_init.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    lib.xmx_window_init.restype = ctypes.c_int
    lib.xmx_rec_gemm_qkv.argtypes = [ctypes.c_int] * 6 + [ctypes.c_uint] * 5
    lib.xmx_rec_gemm_qkv.restype = ctypes.c_int
    lib.xmx_rec_gemm_qkv_window.argtypes = [ctypes.c_int] * 6 + [ctypes.c_uint] * 9
    lib.xmx_rec_gemm_qkv_window.restype = ctypes.c_int
    lib.xmx_rec_gemm_dual.argtypes = [ctypes.c_int] * 4 + [ctypes.c_uint] * 3
    lib.xmx_rec_gemm_dual.restype = ctypes.c_int
    lib.xmx_ffn_init.argtypes = [ctypes.c_char_p]
    lib.xmx_ffn_init.restype = ctypes.c_int
    lib.xmx_rec_ffn.argtypes = [ctypes.c_int] * 6 + [ctypes.c_uint] * 5
    lib.xmx_rec_ffn.restype = ctypes.c_int
    lib.xmx_rec_ffn_merge.argtypes = [ctypes.c_int] * 7 + [ctypes.c_uint] * 4
    lib.xmx_rec_ffn_merge.restype = ctypes.c_int
    lib.xmx_rec_ffn_stem.argtypes = [ctypes.c_int] * 6 + [ctypes.c_uint] * 2
    lib.xmx_rec_ffn_stem.restype = ctypes.c_int
    lib.xmx_rec_gemm_window_residual_pool.argtypes = [ctypes.c_int] * 6 + [ctypes.c_uint] * 8
    lib.xmx_rec_gemm_window_residual_pool.restype = ctypes.c_int
    lib.xmx_window_block_init.argtypes = [ctypes.c_char_p]
    lib.xmx_window_block_init.restype = ctypes.c_int
    lib.xmx_rec_window_block.argtypes = [ctypes.c_int] * 8 + [ctypes.c_uint] * 6
    lib.xmx_rec_window_block.restype = ctypes.c_int
    lib.xmx_global_attention_init.argtypes = [ctypes.c_char_p]
    lib.xmx_global_attention_init.restype = ctypes.c_int
    lib.xmx_rec_global_attention.argtypes = [ctypes.c_int] * 4 + [ctypes.c_uint] * 3
    lib.xmx_rec_global_attention.restype = ctypes.c_int
    lib.xmx_int8_init.argtypes = [ctypes.c_char_p]
    lib.xmx_int8_init.restype = ctypes.c_int
    lib.xmx_rec_gemm_int8.argtypes = [ctypes.c_int] * 5 + [ctypes.c_uint] * 4
    lib.xmx_rec_gemm_int8.restype = ctypes.c_int
    lib.xmx_rec_unary2.argtypes = (
        [ctypes.c_uint]
        + [ctypes.c_int] * 5
        + [ctypes.c_uint, ctypes.c_uint, ctypes.c_float]
        + [ctypes.c_uint] * 5
    )
    lib.xmx_rec_unary2.restype = ctypes.c_int
    lib.xmx_rec_window_attention.argtypes = [ctypes.c_int] * 5 + [ctypes.c_uint] * 3
    lib.xmx_rec_window_attention.restype = ctypes.c_int
    lib.xmx_rec_unary.argtypes = (
        [ctypes.c_uint]
        + [ctypes.c_int] * 4
        + [ctypes.c_uint, ctypes.c_uint, ctypes.c_float]
        + [ctypes.c_uint] * 5
    )
    lib.xmx_rec_unary.restype = ctypes.c_int
    lib.xmx_rec_row.argtypes = (
        [ctypes.c_uint] + [ctypes.c_int] * 4 + [ctypes.c_uint] * 5 + [ctypes.c_float]
    )
    lib.xmx_rec_row.restype = ctypes.c_int
    lib.xmx_rec_qkv.argtypes = [ctypes.c_int] * 5 + [ctypes.c_uint] * 4
    lib.xmx_rec_qkv.restype = ctypes.c_int
    lib.xmx_profile.argtypes = [ctypes.c_int]
    lib.xmx_profile.restype = ctypes.c_int
    lib.xmx_rec_history.argtypes = [ctypes.c_int] * 3 + [ctypes.c_uint] * 5
    lib.xmx_rec_history.restype = ctypes.c_int
    lib.xmx_device_lost.argtypes = []
    lib.xmx_device_lost.restype = ctypes.c_int
    lib.xmx_buf_ptr.argtypes = [ctypes.c_int]
    lib.xmx_buf_ptr.restype = ctypes.c_void_p
    lib.xmx_buf_bytes.argtypes = [ctypes.c_int]
    lib.xmx_buf_bytes.restype = ctypes.c_ulonglong
    lib.xmx_buf_total_bytes.argtypes = []
    lib.xmx_buf_total_bytes.restype = ctypes.c_ulonglong
    lib.xmx_profile_reset.argtypes = []
    lib.xmx_profile_reset.restype = None
    lib.xmx_profile_ms.argtypes = [ctypes.c_uint]
    lib.xmx_profile_ms.restype = ctypes.c_double
    lib.xmx_profile_count.argtypes = [ctypes.c_uint]
    lib.xmx_profile_count.restype = ctypes.c_uint
    lib.xmx_profile_each_count.argtypes = []
    lib.xmx_profile_each_count.restype = ctypes.c_uint
    lib.xmx_profile_each_ms.argtypes = [ctypes.c_uint]
    lib.xmx_profile_each_ms.restype = ctypes.c_double
    lib.xmx_profile_each_kind.argtypes = [ctypes.c_uint]
    lib.xmx_profile_each_kind.restype = ctypes.c_uint
    # The shader paths take an environment override so a variant can be measured
    # against the shipped one without editing the tree. The GEMM kernels follow the
    # device: without VK_KHR_cooperative_matrix (or with XMX_PORTABLE=1) the portable
    # multiply-add builds take the same dispatches, and the staged kernel - which has no
    # portable twin - is never selected by the runtime, so its slot gets the plain one.
    portable = bool(lib.xmx_portable())
    gemm, tiled, staged = (
        ("gemm_portable.spv", "gemm_portable_tiled.spv", "gemm_portable.spv")
        if portable
        else ("gemm_resident.spv", "gemm_tiled.spv", "gemm_staged.spv")
    )
    spv = [
        os.environ.get(name) or nr_build.shader_arg(lib, default)
        for name, default in (
            ("XMX_GEMM_SPV", gemm),
            ("XMX_UNARY_SPV", "resident.spv"),
            ("XMX_ROW_SPV", "attention.spv"),
            ("XMX_HISTORY_SPV", "history.spv"),
            ("XMX_TILED_SPV", tiled),
            ("XMX_STAGED_SPV", staged),
        )
    ]
    if lib.xmx_res_init(*[p.encode() for p in spv]) != 0:
        raise failure(lib, "xmx_res_init")
    # the 32-row staged builds: accepted and not built where there is no staged kernel
    small = [
        os.environ.get(name) or nr_build.shader_arg(lib, default)
        for name, default in (
            ("XMX_STAGED32_SPV", "gemm_staged32.spv"),
            ("XMX_STAGED32_DEEP_SPV", "gemm_staged32_deep.spv"),
        )
    ]
    if lib.xmx_staged32_init(*[p.encode() for p in small]) != 0:
        raise failure(lib, "xmx_staged32_init")
    rows = os.environ.get("XMX_ROWS_SPV") or nr_build.shader_arg(lib, "attention_rows.spv")
    if lib.xmx_rows_init(rows.encode()) != 0:
        raise failure(lib, "xmx_rows_init")
    _lib = lib
    return lib


def backend():
    """The resolved GPU backend name: `vulkan`, `opencl`, `metal` or `d3d12`."""
    return nr_build.backend()


def fused_shader(lib, name):
    """The fused passes' kernel: `name`, or its `_portable` twin on a device without matrix
    units (or under XMX_PORTABLE=1), which sums in the portable GEMM's order and so stays
    bit-identical to the passes it replaces on that path. Embedded when the runtime carries
    it, else the file this build wrote - `nr_build.shader_arg`, as every other shader."""
    if lib.xmx_portable():
        name += "_portable"
    return nr_build.shader_arg(lib, name)
