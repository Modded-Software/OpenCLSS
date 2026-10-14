"""The switches a recording is made under, and the key that cache replays on.

`FuseConfig` is built once from the environment, is immutable afterwards, and is the
only thing besides the device's shader specialisation that `Recorder.graph_key()`
depends on. A switch that changes after a capture but before a replay would replay the
wrong graph without an error anywhere, so the value object is frozen and a variant is a
new object (`FuseConfig.with_config`).
"""

from __future__ import annotations

import os
from collections import namedtuple

from . import native

_FIELDS = (
    "fuse_qk",
    "batch_ffn",
    "input_fp16",
    "compact_head",
    "joint_qkv",
    "fuse_residual",
    "fuse_window_residual",
    "fuse_window_attention",
    "fuse_attention_merge",
    "qkv_epilogue",
    "fuse_glue",
    "fuse_ffn",
    "fuse_branched_ffn",
    "fuse_partition",
    "fuse_transition",
    "fuse_merge_ffn",
    "fuse_stem_ffn",
    "fuse_pool",
    "fuse_window_block",
    "fuse_head",
    "fuse_global_attention",
)


class FuseConfig(namedtuple("FuseConfig", _FIELDS)):
    """The switches, immutable: a variant is a new value from `with_config`."""

    __slots__ = ()
    fuse_qk: bool
    batch_ffn: bool
    input_fp16: bool
    compact_head: bool
    joint_qkv: bool
    fuse_residual: bool
    fuse_window_residual: bool
    fuse_window_attention: bool
    fuse_attention_merge: bool
    qkv_epilogue: bool
    fuse_glue: bool
    fuse_ffn: bool
    fuse_branched_ffn: bool
    fuse_partition: bool
    fuse_transition: bool
    fuse_merge_ffn: bool
    fuse_stem_ffn: bool
    fuse_pool: bool
    fuse_window_block: bool
    fuse_head: bool
    fuse_global_attention: bool

    @classmethod
    def from_environment(cls, lib, backend=None):
        """Every switch, read from the environment once, in one place.

        `backend` is the resolved one (`native.backend()`) unless a caller passes it; it
        decides only `fuse_window_block`, whose default is Vulkan-specific. Two switches
        also ask the device itself (`lib.xmx_window_gather()`), because a lent matrix
        device without the staged kernel cannot gather.
        """
        if backend is None:
            backend = native.backend()
        return cls(
            fuse_qk=os.environ.get("NR_FUSE_QK", "1") != "0",
            batch_ffn=os.environ.get("NR_BATCH_FFN", "1") != "0",
            # On since 2026-09-25: the features are built as half, in the mapped input itself
            # (`ResidentFrame.input_view`), so the GPU's to_half pass and the host's copy both
            # go. The same bytes (test_input_fp16.py); NR_INPUT_FP16=0 is the float32 input.
            input_fp16=os.environ.get("NR_INPUT_FP16", "1") != "0",
            # On since 2026-09-25: the last GEMM stores only the four useful head columns, so
            # the host reads 4 of 16 - 3.6 -> 0.75 ms at 1280x720, 4 ms of the frame, the same
            # bytes (notes/improve-compact-io.md). NR_COMPACT_HEAD=0 is the old sixteen.
            compact_head=os.environ.get("NR_COMPACT_HEAD", "1") != "0",
            joint_qkv=os.environ.get("NR_JOINT_QKV", "0") != "0",
            # ProjectsCodex's fusions (notes/improve-fusions.md). On by default where they
            # were measured exact; each keeps its two-pass path behind a switch.
            fuse_residual=os.environ.get("NR_FUSE_RESIDUAL", "1") != "0",
            fuse_window_residual=os.environ.get("NR_FUSE_WINDOW_RESIDUAL", "1") != "0",
            fuse_window_attention=os.environ.get("NR_FUSE_WINDOW_ATTENTION", "1") != "0",
            # The head merge folded into the fused attention's store: Codex's
            # NR_FUSE_ATTENTION_MERGE, which it left off and unmeasured. Measured here,
            # bit-identical and 3.6-4.4 % of the frame at 720p.
            fuse_attention_merge=os.environ.get("NR_FUSE_ATTENTION_MERGE", "1") != "0",
            # Q/K normalised and V published in the QKV projection's own epilogue, so the
            # float32 projection never goes to memory (qkv_epilogue.glsl). On everywhere: it
            # is 22 % of a 720p frame (`notes/improve-qkv-epilogue.md`), and 143 -> 257 ms of
            # graph at 1344x768 when it is off.
            #
            # Fusing moves the epilogue's writes into the projection's own dispatch, and the
            # projection's input shares a scratch-arena role with `k16`; HANDOFF.md records
            # that the role table was built for passes that finish before the next starts, and
            # this one does not. The epilogue therefore writes K to `key16` instead.
            #
            # This was off on Windows for a while, after horizontal bands on the B580, and
            # that is withdrawn: the bands were the probe resetting the graph's Vulkan state
            # in the daemon's own process (fixed - the probe runs in a child now), and every
            # frame measured since is clean with this on, including with the NumPy host
            # passes. The owner measured the same frame byte-identical with it on and off over
            # 60 frames from three games. `test_gemm_qkv.py` does still differ by the sign of
            # a zero on that driver, with the switch on or off; -0 and +0 are equal in
            # everything downstream, and the test compares bytes deliberately, so it does not
            # implicate the fusion.
            qkv_epilogue=os.environ.get("NR_QKV_EPILOGUE", "1") != "0",
            # The full-resolution glue around blocks 0 and 70 in fewer passes: the stem's
            # GEMM also stores the half copy block 0's feed-forward reads, and block 70's
            # input is upsampled, scaled and merged in one pass that stores both widths.
            fuse_glue=os.environ.get("NR_FUSE_GLUE", "1") != "0",
            # A 32-channel block's feed-forward in one pass, the hidden layer on chip.
            fuse_ffn=os.environ.get("NR_FUSE_FFN", "1") != "0",
            # The branched blocks' per-group expand and projection, the same way.
            fuse_branched_ffn=os.environ.get("NR_FUSE_BRANCHED_FFN", "0") != "0",
            # The window partition folded into the QKV projection's own loads.
            # Off where the runtime cannot gather (a lent matrix device without the staged
            # kernel's extension): the partition is then its own pass, as it always was.
            fuse_partition=(
                os.environ.get("NR_FUSE_PARTITION", "1") != "0" and lib.xmx_window_gather() == 1
            ),
            # A decoder transition's upsample, scaled skip and add in one pass.
            fuse_transition=os.environ.get("NR_FUSE_TRANSITION", "1") != "0",
            # Block 70's input merged inside its fused feed-forward rather than stored twice
            # by a pass of its own for the feed-forward to read back.
            fuse_merge_ffn=os.environ.get("NR_FUSE_MERGE_FFN", "1") != "0",
            # Block 0's stem made inside its fused feed-forward, the same way.
            fuse_stem_ffn=os.environ.get("NR_FUSE_STEM_FFN", "1") != "0",
            # Block 0's output pooled and published in its window residual's own epilogue.
            # Off where the runtime has neither the staged kernel nor the portable 16x32 block
            # to pool in (xmx_window_gather(), the same condition as the partition's).
            fuse_pool=(os.environ.get("NR_FUSE_POOL", "1") != "0" and lib.xmx_window_gather() == 1),
            # A 32-channel window block's QKV projection, attention and output projection in
            # one pass a window, nothing in between leaving the workgroup.
            #
            # Vulkan defaults this off (Run 60, recomp/diary/dlss-nr.md): the fused
            # window_block.comp is the frame's slowest stage per pass (3.7 ms each) because it
            # keeps too much live state per workgroup, and splitting it into the staged QKV
            # GEMM plus the window attention pass is 7-11 ms/frame faster at 1280x768 on the
            # A770, with test_window_block/test_glue/test_joint_qkv/test_determinism all
            # passing. Every other runtime keeps it on: libopenclmx's xmx_rec_window_block is
            # itself the decomposed three-pass path, and Metal/Direct3D have their own fused
            # kernel that is not measured here. NR_FUSE_WINDOW_BLOCK overrides either way.
            fuse_window_block=os.environ.get(
                "NR_FUSE_WINDOW_BLOCK", "0" if backend == "vulkan" else "1"
            )
            != "0",
            # Block 70's output straight into the compact head inside that pass, never stored.
            fuse_head=os.environ.get("NR_FUSE_HEAD", "1") != "0",
            # A bottleneck block's QK^T, softmax, PV and head merge in one pass, no score stored.
            fuse_global_attention=os.environ.get("NR_FUSE_GLOBAL_ATTENTION", "1") != "0",
        )

    def with_config(self, **changes):
        """A new value object with `changes` applied; the receiver is untouched."""
        return self._replace(**changes)

    def key(self, specialization):
        """The graph-replay cache key: the switches, with the device's shader specialisation
        in the low three bits."""
        return (
            specialization
            | (int(self.fuse_qk) << 3)
            | (int(self.batch_ffn) << 4)
            | (int(self.input_fp16) << 5)
            | (int(self.compact_head) << 6)
            | (int(self.joint_qkv) << 7)
            # Bits 8 and up. ProjectsCodex keys these at 5-7, which here are
            # input_fp16, compact_head and joint_qkv: carried over as they were, a
            # graph captured with one setting would replay for the other, and the
            # picture would be wrong without an error anywhere.
            | (int(self.fuse_residual) << 8)
            | (int(self.fuse_window_residual) << 9)
            | (int(self.fuse_window_attention) << 10)
            | (int(self.fuse_attention_merge) << 11)
            | (int(self.qkv_epilogue) << 12)
            | (int(self.fuse_glue) << 13)
            | (int(self.fuse_ffn) << 14)
            | (int(self.fuse_branched_ffn) << 15)
            | (int(self.fuse_partition) << 16)
            | (int(self.fuse_transition) << 17)
            | (int(self.fuse_merge_ffn) << 18)
            | (int(self.fuse_stem_ffn) << 19)
            | (int(self.fuse_pool) << 20)
            | (int(self.fuse_window_block) << 21)
            | (int(self.fuse_head) << 22)
            | (int(self.fuse_global_attention) << 23)
        )

    def scratch_key(self, specialization):
        """What decides which scratch a frame's recording touches: the switches, not the
        shader specialisation (the key's low three bits), so a frame keeps its scratch plan
        and its graphs across a change of specialisation."""
        return self.key(specialization) & ~0x7
