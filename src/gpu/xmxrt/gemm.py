"""GEMM-family pass encoders: dense, residual, QKV, dual, int8, and the fused
feed-forwards. Each is `fn(rec, ...)` and records exactly one `xmx_rec_*` dispatch."""

from __future__ import annotations

import os

from . import ops
from .native import fused_shader, nr_build


def gemm_residual(
    rec,
    a,
    b,
    skip,
    cosine,
    target,
    rows,
    cols,
    inner,
    *,
    epilogue=0,
    narrow=False,
    skip_half=False,
    reverse=None,
):
    """A @ B + skip * cosine, the residual folded into the GEMM's epilogue.

    The two-pass path writes the float32 branch out and reads it back in a residual
    pass of its own; this adds `skip * cosine` before the publish instead, so the
    branch never touches memory and the result is bit-identical
    (`src/gpu/test_gemm_residual.py`). With `reverse=(height, width, 8, origin)` the
    rows are a window block's, padded and in window order, and the output lands
    straight back in the unpadded image. Ported from ProjectsCodex's phases 38-39.
    """
    for extent, multiple, name in (
        (rows, ops.TM, "rows"),
        (cols, ops.TN, "cols"),
        (inner, ops.TK, "inner"),
    ):
        if extent <= 0 or extent % multiple:
            raise ValueError(f"{name}={extent} must be a positive multiple of {multiple}")
    output_rows = rows
    window = ()
    if reverse is not None:
        height, width, size, origin = reverse
        if height <= 0 or width <= 0 or size != 8 or any(not -65535 <= v <= 0 for v in origin):
            raise ValueError(
                "window residual needs positive extents, 8x8 windows and nonpositive origins"
            )
        ph, pw, (top, left) = ops.window_extent(height, width, origin, size)
        if rows != ph * pw:
            raise ValueError("window residual rows must match the padded image")
        output_rows = height * width
        window = (height, width, pw // size, (top << 16) | left)
    if target.id in (a.id, b.id) or target.id == cosine.id:
        raise ValueError("GEMM residual output must not alias matrix inputs or cosine")
    if target.id == skip.id and narrow != skip_half:
        raise ValueError("in-place residual requires matching skip and output widths")
    for buf, size in (
        (a, rows * inner * 2),
        (b, inner * cols * 2),
        (skip, output_rows * cols * (2 if skip_half else 4)),
        (cosine, cols * 4),
        (target, output_rows * cols * (2 if narrow else 4)),
    ):
        if buf.nbytes < size:
            raise ValueError("GEMM residual buffer is too small")
    flags = ops.publish(epilogue, narrow) | (0x40000 if skip_half else 0)
    record = rec.lib.xmx_rec_gemm_window_residual if window else rec.lib.xmx_rec_gemm_residual
    if record(a.id, b.id, target.id, skip.id, cosine.id, rows, cols, inner, flags, *window) != 0:
        raise RuntimeError("xmx_rec_gemm_residual: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def gemm_int8(rec, a, b, c, a_scale, b_scale, rows, cols, inner):
    """C = (A @ B^T) * a_scale[row] * b_scale[column] on the integer path.

    `a` is int8 rows x inner, `b` the weights transposed, int8 cols x inner, the scales
    float32 (`int8_quant.quantise`), `c` float32 (gemm_staged_int8.comp).
    """
    if rows <= 0 or cols <= 0 or cols % 32 or inner <= 0 or inner % 64:
        raise ValueError("the integer GEMM needs cols a multiple of 32 and inner of 64")
    for buf, needed in (
        (a, rows * inner),
        (b, cols * inner),
        (c, rows * cols * 4),
        (a_scale, rows * 4),
        (b_scale, cols * 4),
    ):
        if buf.nbytes < needed:
            raise ValueError("integer GEMM buffer is too small")
    # the runtime builds the `_portable` twin itself where the integer matrix path
    # is missing, so the name is the matrix kernel's either way
    path = os.environ.get("XMX_INT8_SPV") or nr_build.shader_arg(rec.lib, "gemm_staged_int8")
    if rec.lib.xmx_int8_init(path.encode()) != 0:
        raise RuntimeError("integer GEMM pipeline: " + rec.lib.xmx_error().decode())
    if (
        rec.lib.xmx_rec_gemm_int8(a.id, b.id, c.id, a_scale.id, b_scale.id, rows, cols, inner, 0)
        != 0
    ):
        raise RuntimeError("xmx_rec_gemm_int8: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def gemm_residual_pool(
    rec, a, b, skip, cosine, skip_out, pooled, rows, inner, reverse, *, skip_half=False
):
    """Block 0's window residual with both of its readers served from the epilogue.

    What `gemm_residual(..., reverse=reverse)` into a float32 output followed by
    `pool2_skip(output, pooled, skip_out, height, width, 32)` write, bit for bit, without
    the float32 output existing (`src/gpu/test_glue.py`): `skip_out` takes every value
    published as half, `pooled` the published 2x2 pool. 32 channels, even extents and
    pads.
    """
    height, width, size, origin = reverse
    if height <= 0 or width <= 0 or height % 2 or width % 2 or size != 8:
        raise ValueError("a pooled window residual needs even extents and 8x8 windows")
    ph, pw, (top, left) = ops.window_extent(height, width, origin, size)
    if top % 2 or left % 2 or rows != ph * pw:
        raise ValueError("a pooled window residual needs even pads and the padded rows")
    if len({a.id, b.id, skip.id, cosine.id, skip_out.id, pooled.id}) != 6:
        raise ValueError("pooled window residual operands must be distinct")
    pixels = height * width
    for buf, needed in (
        (a, rows * inner * 2),
        (b, inner * 32 * 2),
        (skip, pixels * 32 * (2 if skip_half else 4)),
        (cosine, 32 * 4),
        (skip_out, pixels * 32 * 2),
        (pooled, pixels // 4 * 32 * 2),
    ):
        if buf.nbytes < needed:
            raise ValueError("pooled window residual buffer is too small")
    if (
        rec.lib.xmx_rec_gemm_window_residual_pool(
            a.id,
            b.id,
            skip_out.id,
            skip.id,
            cosine.id,
            pooled.id,
            rows,
            32,
            inner,
            0x40000 if skip_half else 0,
            height,
            width,
            pw // size,
            (top << 16) | left,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_gemm_window_residual_pool: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def gemm_qkv(
    rec,
    a,
    weight,
    q,
    k,
    v,
    scale,
    rows,
    channels,
    heads,
    tokens,
    *,
    window=None,
    image_half=False,
    publish_image=False,
    vit=False,
):
    """The QKV projection, finished in its own epilogue: Q and K cosine-normalised
    (Q times its head's scale), V published, all three as E4M3 halves in
    (window, head, token, 32) order - what the projection into float32 followed by
    two `cosine_publish` and one `split_heads` would write, bit for bit
    (`src/gpu/test_gemm_qkv.py`). `rows` is windows * tokens.

    With `window=(height, width, origin)`, `a` is the image itself (float32, or half
    with `image_half`) and the shifted-window partition happens in the projection's
    own loads: what `partition` into a half buffer then this would write, bit for bit.
    `publish_image` rounds the image to E4M3 on the way in, as a partition with that
    epilogue would. `vit` normalises as the ViT does (`cosine_publish(vit=True)`).
    """
    if (
        min(rows, heads, tokens) <= 0
        or channels != heads * 32
        or rows % tokens
        or rows % ops.TM
        or channels % ops.TK
    ):
        raise ValueError(
            "QKV projection needs 32 channels per head, whole windows and tile-aligned extents"
        )
    if len({q.id, k.id, v.id}) != 3 or {q.id, k.id, v.id} & {a.id, weight.id, scale.id}:
        raise ValueError("QKV targets must be distinct from each other and the inputs")
    if window is not None:
        if vit:
            raise ValueError("the ViT's projection has no windows")
        height, width, origin = window
        ph, pw, (top, left) = ops.window_extent(height, width, origin, 8)
        if tokens != 64 or rows != ph * pw or rows % 64 or channels % 32:
            raise ValueError("a window-gathered QKV projection takes whole 8x8 windows")
        if a.nbytes < height * width * channels * (2 if image_half else 4):
            raise ValueError("QKV projection image is too small")
        for buf in (q, k, v):
            if buf.nbytes < rows * channels * 2:
                raise ValueError("QKV projection buffer is too small")
        if (
            rec.lib.xmx_rec_gemm_qkv_window(
                a.id,
                weight.id,
                q.id,
                k.id,
                v.id,
                scale.id,
                rows,
                channels,
                heads,
                tokens,
                width,
                height,
                pw // 8,
                (top << 16) | left,
                int(image_half) | 2 * int(publish_image),
            )
            != 0
        ):
            raise RuntimeError("xmx_rec_gemm_qkv_window: " + rec.lib.xmx_error().decode())
        rec.note()
        return rec
    if publish_image:
        raise ValueError("only a window-gathered projection publishes its image")
    for buf, size in (
        (a, rows * channels * 2),
        (weight, channels * 3 * channels * 2),
        (q, rows * channels * 2),
        (k, rows * channels * 2),
        (v, rows * channels * 2),
        (scale, heads * 4),
    ):
        if buf.nbytes < size:
            raise ValueError("QKV projection buffer is too small")
    if (
        rec.lib.xmx_rec_gemm_qkv(
            a.id,
            weight.id,
            q.id,
            k.id,
            v.id,
            scale.id,
            rows,
            channels,
            heads,
            tokens,
            int(vit),
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_gemm_qkv: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def gemm_dual(rec, a, b, c, half_copy, rows, cols, inner):
    """C = A @ B in float32, and the same values published into `half_copy`: what a
    GEMM then an `e4m3_half` of its output would write, in one pass - block 0's stem,
    raw for its feed-forward's residual and published for its GEMM."""
    for extent, multiple, name in (
        (rows, ops.TM, "rows"),
        (cols, ops.TN, "cols"),
        (inner, ops.TK, "inner"),
    ):
        if extent <= 0 or extent % multiple:
            raise ValueError(f"{name}={extent} must be a positive multiple of {multiple}")
    if len({a.id, b.id, c.id, half_copy.id}) != 4:
        raise ValueError("GEMM outputs must not alias each other or the inputs")
    for buf, size in (
        (a, rows * inner * 2),
        (b, inner * cols * 2),
        (c, rows * cols * 4),
        (half_copy, rows * cols * 2),
    ):
        if buf.nbytes < size:
            raise ValueError("GEMM buffer is too small")
    if rec.lib.xmx_rec_gemm_dual(a.id, b.id, c.id, half_copy.id, rows, cols, inner) != 0:
        raise RuntimeError("xmx_rec_gemm_dual: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def ffn_fused(
    rec,
    a,
    expand,
    projection,
    target,
    rows,
    channels,
    hidden,
    *,
    groups=1,
    skip=None,
    cosine=None,
    epilogue=0,
    narrow=False,
    skip_half=False,
):
    """A feed-forward in one pass, the hidden layer never leaving the chip.

    With one group of 32 channels and a `skip`, the narrow blocks' whole feed-forward:
    what `gemm(..., EPI_GATE_E4M3, narrow)` into a hidden buffer and `gemm_residual` out
    of it write, bit for bit. With `groups`, the branched blocks' per-group expand and
    projection into `target`'s column slices - what their two grouped GEMMs write -
    and no residual. `src/gpu/test_ffn_fused.py`.
    """
    out = groups * 32
    if (
        channels <= 0
        or channels % 16
        or hidden <= 0
        or hidden % 32
        or rows <= 0
        or rows % 16
        or groups <= 0
    ):
        raise ValueError(
            "fused feed-forward needs channels a multiple of 16, hidden of 32 and 16-row blocks"
        )
    if (skip is None) != (cosine is None) or (skip is not None and (groups != 1 or channels != 32)):
        raise ValueError("the fused residual is for one group of 32 channels")
    inputs = {a.id, expand.id, projection.id} | ({cosine.id} if cosine is not None else set())
    if target.id in inputs or (skip is not None and target.id == skip.id and narrow != skip_half):
        raise ValueError("fused feed-forward output must not alias its inputs")
    sizes = [
        (a, rows * channels * 2),
        (expand, groups * channels * hidden * 2),
        (projection, groups * hidden * 32 * 2),
        (target, rows * out * (2 if narrow else 4)),
    ]
    if skip is not None:
        sizes += [
            (cosine, channels * 4),
            (skip, rows * channels * (2 if skip_half else 4)),
        ]
    if any(buf.nbytes < size for buf, size in sizes):
        raise ValueError("fused feed-forward buffer is too small")
    path = os.environ.get("XMX_FFN_SPV") or fused_shader(rec.lib, "ffn_fused")
    if rec.lib.xmx_ffn_init(path.encode()) != 0:
        raise RuntimeError("fused feed-forward pipeline: " + rec.lib.xmx_error().decode())
    flags = ops.publish(epilogue, narrow) | (0x40000 if skip_half else 0)
    if (
        rec.lib.xmx_rec_ffn(
            a.id,
            expand.id,
            projection.id,
            target.id,
            skip.id if skip is not None else -1,
            cosine.id if cosine is not None else -1,
            rows,
            channels,
            hidden,
            groups,
            flags,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_ffn: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def ffn_fused_merge(
    rec,
    source,
    skip,
    sincos,
    expand,
    projection,
    target,
    cosine,
    height,
    width,
    source_width,
    *,
    epilogue=0,
    narrow=False,
):
    """Block 70's feed-forward with its input made in the same pass.

    What `upsample_merge(source, skip, sincos, merged, merged16, ...)` followed by
    `ffn_fused(merged16, ..., skip=merged, cosine=cosine)` write into `target`, bit for
    bit, without `merged` or `merged16` existing (`src/gpu/test_glue.py`). 32 channels,
    128 hidden; `source` (half) is the level above at half the extent.
    """
    rows = height * width
    if rows <= 0 or rows % 16 or height % 2 or width % 2 or source_width < width // 2:
        raise ValueError("the merged feed-forward needs an even extent in 16-row blocks")
    if target.id in {
        source.id,
        skip.id,
        sincos.id,
        expand.id,
        projection.id,
        cosine.id,
    }:
        raise ValueError("merged feed-forward output must not alias its inputs")
    sizes = [
        (source, (height // 2) * source_width * 32 * 2),
        (skip, rows * 32 * 2),
        (sincos, 64 * 4),
        (expand, 32 * 128 * 2),
        (projection, 128 * 32 * 2),
        (cosine, 32 * 4),
        (target, rows * 32 * (2 if narrow else 4)),
    ]
    if any(buf.nbytes < size for buf, size in sizes):
        raise ValueError("merged feed-forward buffer is too small")
    path = os.environ.get("XMX_FFN_SPV") or fused_shader(rec.lib, "ffn_fused")
    if rec.lib.xmx_ffn_init(path.encode()) != 0:
        raise RuntimeError("fused feed-forward pipeline: " + rec.lib.xmx_error().decode())
    if (
        rec.lib.xmx_rec_ffn_merge(
            source.id,
            skip.id,
            sincos.id,
            expand.id,
            projection.id,
            target.id,
            cosine.id,
            height,
            width,
            source_width,
            ops.publish(epilogue, narrow),
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_ffn_merge: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def ffn_fused_stem(
    rec,
    features,
    adapter,
    expand,
    projection,
    target,
    cosine,
    rows,
    *,
    epilogue=0,
    narrow=False,
):
    """Block 0's feed-forward with its input, the stem, made in the same pass.

    What `gemm_dual(features, adapter, stem, stem16, rows, 32, 16)` followed by
    `ffn_fused(stem16, ..., skip=stem, cosine=cosine)` write into `target`, bit for bit,
    without `stem` or `stem16` existing (`src/gpu/test_glue.py`). `features` is half,
    rows x 16; 32 channels, 128 hidden.
    """
    if rows <= 0 or rows % 16:
        raise ValueError("the stem feed-forward needs 16-row blocks")
    if target.id in {features.id, adapter.id, expand.id, projection.id, cosine.id}:
        raise ValueError("stem feed-forward output must not alias its inputs")
    sizes = [
        (features, rows * 16 * 2),
        (adapter, 16 * 32 * 2),
        (expand, 32 * 128 * 2),
        (projection, 128 * 32 * 2),
        (cosine, 32 * 4),
        (target, rows * 32 * (2 if narrow else 4)),
    ]
    if any(buf.nbytes < size for buf, size in sizes):
        raise ValueError("stem feed-forward buffer is too small")
    path = os.environ.get("XMX_FFN_SPV") or fused_shader(rec.lib, "ffn_fused")
    if rec.lib.xmx_ffn_init(path.encode()) != 0:
        raise RuntimeError("fused feed-forward pipeline: " + rec.lib.xmx_error().decode())
    if (
        rec.lib.xmx_rec_ffn_stem(
            features.id,
            adapter.id,
            expand.id,
            projection.id,
            target.id,
            cosine.id,
            rows,
            ops.publish(epilogue, narrow),
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_ffn_stem: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def gemm(
    rec,
    a,
    b,
    c,
    rows,
    cols,
    inner,
    *,
    batch=1,
    strides=None,
    transpose_b=False,
    leading=None,
    offsets=(0, 0, 0),
    epilogue=0,
    narrow=False,
    compact_output=False,
):
    """C = A @ B for tile-aligned extents; A and B are float16, C float32.

    `strides` are element counts per batch item, taken as the dense packing when omitted;
    `leading` overrides the row strides of A, B and C, and `offsets` shifts each
    operand's base in elements, so a GEMM can read or write a slice of a wider
    buffer - which is how the branched and split feed-forwards place their heads.
    `compact_output` stores only the first four of sixteen columns in FP32,
    using the base kernel's shared-memory scatter. Batch output stride defaults
    to rows*4; custom leading/offsets and epilogues are deliberately excluded.
    Extents must already be multiples of 8 / 16 / 16: cooperative-matrix loads are
    not bounds-checked on this device (`cooperativeMatrixRobustBufferAccess` is
    false), so the padding has to be in the buffer, not in a guard.
    """
    for extent, multiple, name in (
        (rows, ops.TM, "rows"),
        (cols, ops.TN, "cols"),
        (inner, ops.TK, "inner"),
    ):
        if extent % multiple:
            raise ValueError(f"{name}={extent} must be a multiple of {multiple}")
    if compact_output:
        if cols != 16 or narrow or epilogue or leading is not None or offsets != (0, 0, 0):
            raise ValueError(
                "compact output requires 16 columns, plain FP32 and no custom leading/offsets"
            )
        leading = (0, 0, 4)
    if strides is None:
        strides = (
            rows * inner,
            cols * inner if transpose_b else inner * cols,
            rows * (4 if compact_output else cols),
        )
    lda, ldb, ldc = leading or (0, 0, 0)
    flags = (
        (1 if transpose_b else 0)
        | ops.publish(epilogue, narrow)
        | (0x10000 if compact_output else 0)
    )
    if (
        rec.lib.xmx_rec_gemm(
            a.id,
            b.id,
            c.id,
            rows,
            cols,
            inner,
            batch,
            strides[0],
            strides[1],
            strides[2],
            flags,
            lda,
            ldb,
            ldc,
            *offsets,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_gemm: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec
