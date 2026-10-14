"""Element-wise, layout and glue pass encoders, on the shared `unary` dispatch.

`unary` is the one entry point most of these build on; the layout passes (partition,
reverse, split/merge heads) and the pooling and upsampling passes are the rest. Each is
`fn(rec, ...)`.
"""

from __future__ import annotations

from . import ops


def unary(
    rec,
    kind,
    source,
    target,
    count,
    *,
    scale=1.0,
    second=None,
    third=None,
    channels=0,
    epilogue=0,
    narrow=False,
    a_half=False,
    b_half=False,
    _dims=None,
    _pad=0,
):
    second = second if second is not None else source
    third = third if third is not None else source
    batch, height, width, across = _dims or (0, 0, 0, 0)
    if (
        rec.lib.xmx_rec_unary(
            kind | ops.publish(epilogue, narrow) | ops.reads(a_half, b_half),
            source.id,
            second.id,
            target.id,
            third.id,
            int(count),
            int(channels),
            float(scale),
            int(batch),
            int(height),
            int(width),
            int(across),
            int(_pad),
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_unary: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def e4m3(rec, source, target, count, scale=1.0):
    return unary(rec, ops.E4M3, source, target, count, scale=scale)


def gate(rec, source, target, count, scale=1.0):
    return unary(rec, ops.GATE, source, target, count, scale=scale)


def half(rec, source, target, count, scale=1.0):
    return unary(rec, ops.HALF, source, target, count, scale=scale)


def gate_e4m3_half(rec, source, target, count):
    """gate, publish and narrow in one read and one write.

    The graph writes a float32 hidden buffer, gates it, publishes it and narrows
    it to half - four trips over 503 MB at 720p in block 0 alone. This is one.
    """
    return unary(rec, ops.GATE_E4M3_HALF, source, target, count)


def e4m3_half(rec, source, target, count):
    return unary(rec, ops.E4M3_HALF, source, target, count)


def gate_half(rec, source, target, count):
    return unary(rec, ops.GATE_HALF, source, target, count)


def to_half(rec, source, target, count, scale=1.0):
    """float32 -> float16, for a GEMM operand."""
    return unary(rec, ops.TO_HALF, source, target, count, scale=scale)


def from_half(rec, source, target, count, scale=1.0):
    return unary(rec, ops.FROM_HALF, source, target, count, scale=scale)


def scale(rec, source, target, count, factor):
    return unary(rec, ops.SCALE, source, target, count, scale=factor)


def residual(
    rec,
    branch,
    skip,
    cosine,
    target,
    count,
    channels,
    *,
    reverse=None,
    epilogue=0,
    narrow=False,
    a_half=False,
    b_half=False,
):
    """target = branch + skip * cosine, one cosine per channel."""
    if reverse is not None:
        height, width, size, origin = reverse
        _, pw, (top, left) = ops.window_extent(height, width, origin, size)
        return unary(
            rec,
            ops.RESIDUAL | 0x4000,
            branch,
            target,
            count,
            second=skip,
            third=cosine,
            channels=channels,
            epilogue=epilogue,
            narrow=narrow,
            a_half=a_half,
            b_half=b_half,
            _dims=(size, height, width, pw // size),
            _pad=(top << 16) | left,
        )
    return unary(
        rec,
        ops.RESIDUAL,
        branch,
        target,
        count,
        second=skip,
        third=cosine,
        channels=channels,
        epilogue=epilogue,
        narrow=narrow,
        a_half=a_half,
        b_half=b_half,
    )


def partition(
    rec,
    source,
    target,
    height,
    width,
    channels,
    size=8,
    origin=(0, 0),
    *,
    epilogue=0,
    narrow=False,
    a_half=False,
):
    """NHWC -> (windows, tokens, channels), the shifted-window origin folded in.

    The vendor pads by up to one window before partitioning; rather than write that
    padded copy, the gather reads zero outside the image.
    """
    ph, pw, (top, left) = ops.window_extent(height, width, origin, size)
    return unary(
        rec,
        ops.PARTITION,
        source,
        target,
        ph * pw * channels,
        channels=channels,
        scale=1.0,
        epilogue=epilogue,
        narrow=narrow,
        a_half=a_half,
        _dims=(size, height, width, pw // size),
        _pad=(top << 16) | left,
    )


def reverse(rec, source, target, height, width, channels, size=8, origin=(0, 0)):
    """(windows, tokens, channels) -> NHWC, cropping the shifted-window pad away."""
    _ph, pw, (top, left) = ops.window_extent(height, width, origin, size)
    return unary(
        rec,
        ops.REVERSE,
        source,
        target,
        height * width * channels,
        channels=channels,
        scale=1.0,
        _dims=(size, height, width, pw // size),
        _pad=(top << 16) | left,
    )


def split_heads(
    rec,
    source,
    target,
    windows,
    tokens,
    channels,
    heads,
    part,
    *,
    epilogue=0,
    narrow=False,
):
    """(windows, tokens, 3C) -> Q, K or V as (windows, heads, tokens, 32)."""
    return unary(
        rec,
        ops.SPLIT_HEADS,
        source,
        target,
        windows * tokens * channels,
        channels=channels,
        epilogue=epilogue,
        narrow=narrow,
        _dims=(heads, tokens, part, 0),
    )


def merge_heads(
    rec,
    source,
    target,
    windows,
    tokens,
    channels,
    heads,
    *,
    epilogue=0,
    narrow=False,
    scale=None,
):
    """(windows, heads, tokens, 32) -> (windows, tokens, C). With `scale` - a
    reciprocal a (window, head, token) row, as `vit_softmax` writes them - each value is
    rounded to half and multiplied by its row's: the ViT's normalisation."""
    if scale is None:
        return unary(
            rec,
            ops.MERGE_HEADS,
            source,
            target,
            windows * tokens * channels,
            channels=channels,
            epilogue=epilogue,
            narrow=narrow,
            _dims=(heads, tokens, 0, 0),
        )
    if scale.nbytes < windows * heads * tokens * 4 or scale.id in (
        source.id,
        target.id,
    ):
        raise ValueError("the head merge's reciprocals need a buffer of their own")
    return unary(
        rec,
        ops.MERGE_HEADS | 0x20000,
        source,
        target,
        windows * tokens * channels,
        channels=channels,
        third=scale,
        epilogue=epilogue,
        narrow=narrow,
        _dims=(heads, tokens, 0, 0),
    )


def pool2_skip(rec, source, pooled, skip, height, width, channels):
    """`pool2` with the E4M3 publish into half `pooled`, and `e4m3_half` of the same
    float32 `source` into `skip`, in one read of it (`src/gpu/test_glue.py`). Even
    extents only: an odd last row or column would be pooled away and never skipped."""
    count = (height // 2) * (width // 2) * channels
    if height % 2 or width % 2:
        raise ValueError("pool2_skip needs even extents")
    if len({source.id, pooled.id, skip.id}) != 3:
        raise ValueError("pool2_skip operands must be distinct")
    for buf, size in (
        (source, height * width * channels * 4),
        (pooled, count * 2),
        (skip, height * width * channels * 2),
    ):
        if buf.nbytes < size:
            raise ValueError("pool2_skip buffer is too small")
    if (
        rec.lib.xmx_rec_unary2(
            ops.POOL2_SKIP | ops.publish(ops.EPI_E4M3, True),
            source.id,
            source.id,
            pooled.id,
            source.id,
            skip.id,
            int(count),
            int(channels),
            1.0,
            0,
            int(height),
            int(width),
            0,
            0,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_unary2: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def pool2(
    rec,
    source,
    target,
    height,
    width,
    channels,
    *,
    epilogue=0,
    narrow=False,
    a_half=False,
):
    """2x2 average pool, NHWC."""
    return unary(
        rec,
        ops.POOL2,
        source,
        target,
        (height // 2) * (width // 2) * channels,
        channels=channels,
        epilogue=epilogue,
        narrow=narrow,
        a_half=a_half,
        _dims=(0, height, width, 0),
    )


def upsample2(
    rec,
    source,
    target,
    source_width,
    height,
    width,
    channels,
    *,
    a_half=False,
    narrow=False,
):
    """Nearest 2x upsample, cropped to (height, width)."""
    return unary(
        rec,
        ops.UPSAMPLE2,
        source,
        target,
        height * width * channels,
        channels=channels,
        a_half=a_half,
        narrow=narrow,
        _dims=(0, width, source_width, 0),
    )


def pad_end(
    rec,
    source,
    target,
    height,
    width,
    padded_height,
    padded_width,
    channels,
    *,
    a_half=False,
    narrow=False,
):
    """Extend to a larger extent with zeros, as `pad_spatial_end` does."""
    return unary(
        rec,
        ops.PAD_END,
        source,
        target,
        padded_height * padded_width * channels,
        channels=channels,
        a_half=a_half,
        narrow=narrow,
        _dims=(0, height, width, padded_width),
    )


def upsample_add(
    rec,
    source,
    skip,
    factors,
    target,
    height,
    width,
    source_width,
    channels,
    *,
    skip_half=False,
    epilogue=0,
    narrow=False,
    published=None,
):
    """target = upsample2(source) + skip * factors, published: what upsample2, then
    scale_channel of the skip, then add with the epilogue write, in one pass
    (`src/gpu/test_glue.py`). `source` is float32, cropped to (height, width).
    `published` takes the same values published to E4M3 as halves, besides `target`."""
    if published is None:
        return unary(
            rec,
            ops.UPSAMPLE_ADD,
            source,
            target,
            height * width * channels,
            channels=channels,
            second=skip,
            third=factors,
            epilogue=epilogue,
            narrow=narrow,
            b_half=skip_half,
            _dims=(0, width, source_width, 0),
        )
    count = height * width * channels
    if published.id in {source.id, skip.id, factors.id, target.id} or published.nbytes < count * 2:
        raise ValueError("the published merge needs a buffer of its own")
    if (
        rec.lib.xmx_rec_unary2(
            ops.UPSAMPLE_ADD | ops.publish(epilogue, narrow) | ops.reads(False, skip_half),
            source.id,
            skip.id,
            target.id,
            factors.id,
            published.id,
            int(count),
            int(channels),
            1.0,
            0,
            int(width),
            int(source_width),
            0,
            0,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_unary2: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def upsample_merge(
    rec, source, skip, sincos, merged, merged16, height, width, source_width, channels
):
    """merged = upsample2(source) * sin + skip * cos, stored float32 and published.

    `source` (half) is the level above at half the extent, `skip` (half) the
    full-resolution skip, `sincos` the per-channel sin then cos. The same values the
    upsample2, scale_channel, residual and e4m3_half passes produce, in one pass
    (`src/gpu/test_glue.py`).
    """
    count = height * width * channels
    if len({source.id, skip.id, sincos.id, merged.id, merged16.id}) != 5:
        raise ValueError("upsample merge operands must be distinct")
    for buf, size in (
        (skip, count * 2),
        (sincos, channels * 8),
        (merged, count * 4),
        (merged16, count * 2),
        (source, -(-height // 2) * source_width * channels * 2),
    ):
        if buf.nbytes < size:
            raise ValueError("upsample merge buffer is too small")
    if (
        rec.lib.xmx_rec_unary2(
            ops.UPSAMPLE_MERGE | ops.reads(True, True),
            source.id,
            skip.id,
            merged.id,
            sincos.id,
            merged16.id,
            int(count),
            int(channels),
            1.0,
            0,
            int(width),
            int(source_width),
            0,
            0,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_unary2: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec


def scale_channel(rec, source, factors, target, count, channels, *, a_half=False):
    """target = source * factors, one factor per channel."""
    return unary(
        rec,
        ops.SCALE_CHANNEL,
        source,
        target,
        count,
        channels=channels,
        third=factors,
        a_half=a_half,
    )


def add(
    rec,
    left,
    right,
    target,
    count,
    *,
    epilogue=0,
    narrow=False,
    a_half=False,
    b_half=False,
):
    return unary(
        rec,
        ops.ADD,
        left,
        target,
        count,
        second=right,
        epilogue=epilogue,
        narrow=narrow,
        a_half=a_half,
        b_half=b_half,
    )


def add_bias(rec, source, bias, target, count, tokens, heads):
    """scores + the per-head attention bias."""
    return unary(
        rec,
        ops.ADD_BIAS,
        source,
        target,
        count,
        channels=tokens,
        third=bias,
        _dims=(heads, 0, 0, 0),
    )


def sample_history(rec, history, motion, target, height, width, channels=3, absolute=False):
    """Reproject `history` with the recovered five-tap Catmull-Rom filter.

    `motion` holds either the offsets, or the sample coordinates themselves when
    `absolute`, which is the form `sample_history` in the reference takes.
    """
    if (
        rec.lib.xmx_rec_history(
            history.id,
            motion.id,
            target.id,
            height * width,
            channels,
            height,
            width,
            1 if absolute else 0,
        )
        != 0
    ):
        raise RuntimeError("xmx_rec_history: " + rec.lib.xmx_error().decode())
    rec.note()
    return rec
