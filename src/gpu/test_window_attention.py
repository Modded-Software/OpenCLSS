#!/usr/bin/env python3
"""Fused window attention against the three-pass path it replaces, on identical inputs.

ProjectsCodex's test, in both output layouts: the FP32 context, and the merged one that
also does `merge_heads`' work (`fuse_attention_merge`).
"""

import numpy as np
from xmxrt import attention, elementwise, gemm, memory, ops
from xmxrt.recorder import Recorder


def main():
    rt = Recorder()
    rng = np.random.default_rng(42)
    cases = 0
    for batches, heads in ((1, 1), (6, 3), (32, 16)):
        buffers = []

        def alloc(count, dtype=np.float32):
            buf = rt.buffer(count, dtype)
            buffers.append(buf)
            return buf

        try:
            q, k, v = [alloc(batches * 2048, np.float16) for _ in range(3)]
            bias = alloc(heads * 4096)
            scores = alloc(batches * 4096)
            probs = alloc(batches * 4096, np.float16)
            expected, actual = [alloc(batches * 2048 + 32) for _ in range(2)]
            expected_merge, actual_merge = [
                alloc(batches * 2048 + 32, np.float16) for _ in range(2)
            ]
            for scale in (0.01, 0.3, 5.0, None):
                for buf in (q, k, v):
                    values = rng.normal(0, scale or 0.3, batches * 2048).astype(np.float16)
                    values[:8] = [0, -0.0, 2**-20, -(2**-20), 0.125, -0.25, 1, -2]
                    memory.host_write(buf, values)
                bias_values = rng.uniform(-7, 7, heads * 4096).astype(np.float32)
                if scale is None:
                    # Zero scores expose every finite-half bias to the bit-affine
                    # transform, including clamp boundaries and half overflow in sums.
                    for buf in (q, k):
                        memory.host_write(buf, np.zeros(batches * 2048, np.float16))
                    finite = np.arange(65536, dtype=np.uint16).view(np.float16)
                    finite = finite[np.isfinite(finite)].astype(np.float32)
                    bias_values = np.resize(finite, heads * 4096)
                memory.host_write(bias, bias_values)
                for use_bias in (False, True):
                    selected = bias if use_bias else None
                    for mask in (0, 7):
                        rt.specialize(mask)
                        for buf in (expected, actual):
                            memory.host_write(buf, np.full(batches * 2048 + 32, -11, np.float32))
                        for buf in (expected_merge, actual_merge):
                            memory.host_write(buf, np.full(batches * 2048 + 32, -11, np.float16))
                        rt.begin()
                        gemm.gemm(
                            rt,
                            q,
                            k,
                            scores,
                            64,
                            64,
                            32,
                            batch=batches,
                            transpose_b=True,
                        )
                        attention.softmax(
                            rt,
                            scores,
                            probs,
                            batches * 64,
                            64,
                            narrow=True,
                            bias=selected,
                            heads=heads,
                        )
                        gemm.gemm(rt, probs, v, expected, 64, 32, 64, batch=batches)
                        elementwise.merge_heads(
                            rt,
                            expected,
                            expected_merge,
                            batches // heads,
                            64,
                            heads * 32,
                            heads,
                            epilogue=ops.EPI_E4M3,
                            narrow=True,
                        )
                        attention.window_attention(
                            rt, q, k, v, actual, batches, heads, bias=selected
                        )
                        attention.window_attention(
                            rt,
                            q,
                            k,
                            v,
                            actual_merge,
                            batches,
                            heads,
                            bias=selected,
                            merged=True,
                        )
                        rt.submit()
                        np.testing.assert_array_equal(
                            memory.host_view(actual).view(np.uint8),
                            memory.host_view(expected).view(np.uint8),
                        )
                        # the 32 guard values past the end prove the merged store stays
                        # inside its (window, token, C) extent
                        np.testing.assert_array_equal(
                            memory.host_view(actual_merge, np.float16).view(np.uint8),
                            memory.host_view(expected_merge, np.float16).view(np.uint8),
                        )
                        cases += 1
            for target, b, h in (
                (q, batches, heads),
                (actual, 0, heads),
                (actual, batches, 0),
                (actual, 2**32, 1),
            ):
                try:
                    attention.window_attention(rt, q, k, v, target, b, h, bias=bias)
                except ValueError:
                    rejected = True
                else:
                    rejected = False
                assert rejected, "invalid attention accepted"
            tiny = alloc(1)
            try:
                attention.window_attention(rt, q, k, v, tiny, batches, heads, bias=bias)
            except ValueError:
                rejected = True
            else:
                rejected = False
            assert rejected, "undersized output accepted"
        finally:
            for buf in buffers:
                buf.free()
    for name in ("fuse_window_attention", "fuse_attention_merge"):
        old = rt.graph_key()
        rt.with_config(**{name: not getattr(rt.config, name)})
        assert old != rt.graph_key(), name
    print(
        f"window attention: {cases} bit-exact cases in both output layouts, "
        f"guards and graph keys OK"
    )


if __name__ == "__main__":
    main()
