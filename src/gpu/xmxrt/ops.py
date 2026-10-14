"""Op kinds, epilogue codes and the flag builders every pass shares.

The values are the ABI with the C side (`xmx.h`, `libopenclmx.c`): a kind, an epilogue
code or a flag bit here is a number the kernels read, so none of them may move.
"""

TM, TN, TK = 8, 16, 16

(
    E4M3,
    GATE,
    HALF,
    TO_HALF,
    SCALE,
    RESIDUAL,
    FROM_HALF,
    PARTITION,
    REVERSE,
    ADD_BIAS,
    SPLIT_HEADS,
    MERGE_HEADS,
    POOL2,
    UPSAMPLE2,
    SCALE_CHANNEL,
    ADD,
    PAD_END,
    GATE_E4M3_HALF,
    E4M3_HALF,
    GATE_HALF,
    UPSAMPLE_MERGE,
    UPSAMPLE_ADD,
    POOL2_SKIP,
) = range(23)
COSINE_PUBLISH, SOFTMAX, VIT_SOFTMAX = 0, 1, 6

# GEMM epilogues, applied to the accumulator on its way out of the kernel
EPI_NONE, EPI_E4M3, EPI_GATE, EPI_GATE_E4M3, EPI_HALF = 0, 1, 2, 3, 4

# what a buffer is for, which decides where it lives
GRAPH, HOST_READ, HOST_WRITE = 0, 1, 2


def reads(a_half=False, b_half=False):
    """Operand width bits: 15 marks `a` as float16, 16 marks `b`."""
    return (0x8000 if a_half else 0) | (0x10000 if b_half else 0)


def publish(epilogue, narrow):
    """The publish half of a pass's flags: bits 8-11 the epilogue, bit 12 a half output.

    Every shader reads the same layout, so a pass that ends in a publish never needs a
    second dispatch over the same buffer to apply it.
    """
    return (int(epilogue) << 8) | (0x1000 if narrow else 0)


def align(value, multiple):
    return -(-value // multiple) * multiple


def window_extent(height, width, origin=(0, 0), size=8):
    """-> (padded height, padded width, pads) for a shifted-window partition."""
    pad_top, pad_left = -origin[0], -origin[1]
    padded_height = pad_top + height + (-(height + pad_top)) % size
    padded_width = pad_left + width + (-(width + pad_left)) % size
    return padded_height, padded_width, (pad_top, pad_left)
