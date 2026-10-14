"""Attention-family pass encoders: window attention and block, global attention, the
two softmaxes and the cosine publish. Each is `fn(rec, ...)`."""

from __future__ import annotations

import os

from . import ops
from .native import fused_shader


def window_block(
    rec,
    image,
    qkv,
    projection,
    target,
    bias,
    cosine,
    scale,
    height,
    width,
    origin,
    *,
    epilogue=0,
    narrow=False,
    image_half=False,
    pooled=None,
    head=None,
    head_columns=4,
):
    """A 32-channel window block's attention half in one pass (window_block.comp).

    What `gemm_qkv(image, qkv, ..., window=(height, width, origin))`, `window_attention(
    ..., merged=True)` and `gemm_residual(attended, projection, image, cosine, target,
    ..., reverse=(height, width, 8, origin))` write into `target`, bit for bit, with
    nothing in between leaving the workgroup (`src/gpu/test_window_block.py`). `image`
    is the projection's input and the residual's skip, float32 or half; one head.

    With `pooled`, block 0's output as `gemm_residual_pool` writes it: `target` takes the
    published skip (half), `pooled` the published 2x2 pool; even extents and pads.

    With `head` (the head's padded 32 x 16 weights), block 70's: its output is not stored
    and `target` takes what `gemm(output16, head, target, pixels, 16, 32,
    compact_output=True)` would, four float32 columns a pixel - or with
    `head_columns=16` what the same GEMM stores without the compact output.
    """
    ph, pw, (top, left) = ops.window_extent(height, width, origin, 8)
    if pooled is not None:
        if height % 2 or width % 2 or top % 2 or left % 2 or epilogue or not narrow:
            raise ValueError(
                "a pooled window block needs even extents and pads, no publish and a half target"
            )
        if pooled.nbytes < height * width // 4 * 32 * 2 or pooled.id == target.id:
            raise ValueError("the pooled output is too small or aliases the target")
    windows = (ph // 8) * (pw // 8)
    if height <= 0 or width <= 0 or not windows:
        raise ValueError("a window block needs a positive extent")
    if target.id in {image.id, qkv.id, projection.id, bias.id, cosine.id, scale.id}:
        raise ValueError("the window block's target must not alias its inputs")
    pixels = height * width
    if head is not None and (pooled is not None or epilogue or narrow):
        raise ValueError("the head is block 70's: no pool, no publish, a float32 target")
    if head_columns not in (4, 16):
        raise ValueError("the head stores four columns or all sixteen")
    output = pixels * head_columns * 4 if head is not None else pixels * 32 * (2 if narrow else 4)
    for buf, needed in (
        (image, pixels * 32 * (2 if image_half else 4)),
        (qkv, 32 * 96 * 2),
        (projection, 32 * 32 * 2),
        (bias, 64 * 64 * 4),
        (cosine, 32 * 4),
        (scale, 4),
        (target, output),
    ) + (((head, 32 * 16 * 2),) if head else ()):
        if buf.nbytes < needed:
            raise ValueError("window block buffer is too small")
    path = os.environ.get("XMX_WINDOW_BLOCK_SPV") or fused_shader(rec.lib, "window_block")
    if rec.lib.xmx_window_block_init(path.encode()) != 0:
        raise RuntimeError("window block pipeline: " + rec.lib.xmx_error().decode())
    flags = (
        ops.publish(epilogue, narrow)
        | (0x8000 if image_half else 0)
        | (0x800000 if pooled is not None else 0)
        | (0x1000000 if head is not None else 0)
        | (0x2000000 if head is not None and head_columns == 16 else 0)
    )
    if pooled is not None:
        flags &= ~0x1000  # the pool's target is half by definition
    extra = pooled if pooled is not None else head
    if (
        rec.lib.xmx_rec_window_block(
            image.id,
            qkv.id,
            projection.id,
            target.id,
            bias.id,
            cosine.id,
            scale.id,
            extra.id if extra is not None else -1,
            windows,
            height,
            width,
            pw // 8,
            (top << 16) | left,
            flags,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_window_block: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def window_attention(rec, q, k, v, target, batches, heads, *, bias=None, merged=False):
    """QK^T, softmax and PV for full 8x8 windows, in one dispatch, into `target`.

    The scores and probabilities never leave shared memory. The result is the same
    FP32 context the three-pass path writes, bit for bit
    (`src/gpu/test_window_attention.py`). ProjectsCodex's phase42.

    `merged` does the head merge in the same dispatch: `target` then receives what
    `merge_heads(context, ..., epilogue=EPI_E4M3, narrow=True)` would have written -
    float16 E4M3 values in (window, token, head, channel) order - also bit for bit.
    """
    if (
        batches <= 0
        or heads <= 0
        or batches % heads
        or batches * 2048 > 0xFFFFFFFF
        or heads * 4096 > 0xFFFFFFFF
    ):
        raise ValueError("invalid window attention batch/head count")
    operands = (q, k, v) + ((bias,) if bias is not None else ())
    if any(target.id == buf.id for buf in operands):
        raise ValueError("window attention output must not alias an input")
    sizes = [
        (q, batches * 4096),
        (k, batches * 4096),
        (v, batches * 4096),
        (target, batches * (4096 if merged else 8192)),
    ]
    if bias is not None:
        sizes.append((bias, heads * 4096 * 4))
    if any(buf.nbytes < size for buf, size in sizes):
        raise ValueError("window attention buffer is too small")
    path = os.environ.get("XMX_WINDOW_SPV") or fused_shader(rec.lib, "window_attention")
    if rec.lib.xmx_window_init(path.encode(), int(merged)) != 0:
        raise RuntimeError("window attention pipeline: " + rec.lib.xmx_error().decode())
    if (
        rec.lib.xmx_rec_window_attention(
            q.id,
            k.id,
            v.id,
            bias.id if bias is not None else -1,
            target.id,
            batches,
            heads,
            int(merged),
        )
        != 0
    ):
        raise RuntimeError("window attention: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def global_attention(rec, q, k, v, merged, rows, tokens, heads):
    """A bottleneck block's attention in one pass (global_attention.comp).

    What `gemm(q, k, scores, rows, rows, 32, batch=heads, transpose_b=True)`,
    `vit_softmax(scores, probs, reciprocal, heads * rows, tokens, stride=rows)`,
    `gemm(probs, v, context, ...)` and `merge_heads(context, merged, 1, rows, 32 * heads,
    heads, epilogue=EPI_E4M3, narrow=True, scale=reciprocal)` write into `merged`, bit
    for bit, with no score stored (`src/gpu/test_global_attention.py`).
    """
    if (
        rows <= 0
        or rows % 16
        or not 0 < tokens <= rows
        or heads <= 0
        or -(-rows // 64) != -(-tokens // 64)
    ):
        raise ValueError("global attention needs rows a multiple of 16 within the tokens' 64-block")
    if merged.id in {q.id, k.id, v.id}:
        raise ValueError("global attention's output must not alias its inputs")
    for buf in (q, k, v, merged):
        if buf.nbytes < heads * rows * 32 * 2:
            raise ValueError("global attention buffer is too small")
    path = os.environ.get("XMX_GLOBAL_ATTENTION_SPV") or fused_shader(rec.lib, "global_attention")
    if rec.lib.xmx_global_attention_init(path.encode()) != 0:
        raise RuntimeError("global attention pipeline: " + rec.lib.xmx_error().decode())
    if rec.lib.xmx_rec_global_attention(q.id, k.id, v.id, merged.id, rows, tokens, heads) != 0:
        raise RuntimeError("xmx_rec_global_attention: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def cosine_publish(
    rec,
    source,
    target,
    rows,
    *,
    tokens=0,
    heads=0,
    scale=None,
    narrow=False,
    from_half=False,
    qkv_part=None,
    vit=False,
):
    """Normalise rows of 32 through the kernel's fragment tree, then publish as E4M3.

    With `scale` the query path also multiplies by its head's `attn_scale`; rows are
    ordered (batch, head, token), so the head follows from the row index. `vit` takes
    the ViT's tree instead, and its query half(sqrt(32)) before the scale.
    """
    third = scale if scale is not None else source
    flags = (
        ops.COSINE_PUBLISH
        | ops.publish(0, narrow)
        | (0x8000 if from_half else 0)
        | (0x80000 if vit else 0)
    )
    if qkv_part is not None:
        if qkv_part not in (0, 1) or tokens <= 0 or heads <= 0 or from_half:
            raise ValueError("QKV gather needs part 0/1, positive tokens/heads and float32 input")
        flags |= 0x20000 | (qkv_part << 18)
    if (
        rec.lib.xmx_rec_row(
            flags,
            source.id,
            source.id,
            target.id,
            third.id,
            int(rows),
            int(tokens),
            int(heads),
            1 if scale is not None else 0,
            0,
            0.0,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_row: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def vit_softmax(rec, source, target, reciprocal, rows, tokens, *, stride=0, narrow=True):
    """The ViT's softmax (attention.comp, `vit_softmax`): each row's weights
    published unnormalised into `target`, and its reciprocal, the vendor's denominator,
    into `reciprocal` (float32, one a row) for `merge_heads(scale=...)` to apply."""
    width = stride or tokens
    if rows <= 0 or not 0 < tokens <= width:
        raise ValueError("the ViT softmax needs rows and at least one token a row")
    if reciprocal.nbytes < rows * 4 or reciprocal.id in (source.id, target.id):
        raise ValueError("the ViT softmax's reciprocals need a buffer of their own")
    if (
        rec.lib.xmx_rec_row(
            ops.VIT_SOFTMAX | ops.publish(0, narrow),
            source.id,
            source.id,
            target.id,
            reciprocal.id,
            int(rows),
            int(tokens),
            0,
            0,
            int(stride),
            0.0,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_row: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def softmax(
    rec,
    source,
    target,
    rows,
    width,
    *,
    stride=0,
    cap=0.0,
    narrow=False,
    bias=None,
    heads=0,
):
    """The bit-affine softmax, one row per invocation.

    `stride` lets a row be wider than its token count, which the global blocks
    need: their token count is the bottleneck's pixel count and need not be a
    multiple of the tile. `cap` is the symmetric logit clamp the vit_1d kernels
    apply.
    """
    flags = ops.SOFTMAX | ops.publish(0, narrow) | (0x2000 if bias is not None else 0)
    if (
        rec.lib.xmx_rec_row(
            flags,
            source.id,
            (bias or source).id,
            target.id,
            source.id,
            int(rows),
            int(width),
            int(heads),
            0,
            int(stride),
            float(cap),
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_row: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec
