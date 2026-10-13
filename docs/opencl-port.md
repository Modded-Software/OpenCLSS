# Porting the frame to OpenCL (Intel DG2 / Arc A-series)

Status: **plan**, 2026-10-13. Target: the 1024x576 network field at 24-30 fps (33-41 ms
a frame) on one A770, measured in the frame, not in a microbenchmark.

## 1. Why

The Vulkan frame is 82.3 ms of device time (1024x576, `frame_profile`, Run 26):

| pass family | ms | passes | share |
|---|---:|---:|---:|
| GEMM (staged, all epilogues) | 41.4 | 318 | 50 % |
| window block (fused C=32 attention half) | 22.5 | 10 | 27 % |
| ffn fused (+ merge, stem) | 8.5 | 10 | 10 % |
| window attention | 8.0 | 52 | 10 % |
| global attention | 1.1 | 8 | 1 % |
| unary (upsample add, e4m3 half, pool2, pad) | 0.7 | 17 | 1 % |

All of it runs at 4-10 % of the card. Mesa's cooperative-matrix lowering on DG2 turns
every tile load into scalar SLM or global messages (~650 instructions per 4 DPAS in the
staged K loop, `notes/phase77`), and Vulkan has no way to ask for anything else: ANV
exposes no Intel subgroup block-IO extension (`vulkaninfo`: only
`VK_INTEL_shader_integer_functions2`). Every kernel tweak inside that ceiling has moved
the frame by +-10 % (Runs 17, 24, 26, 27).

Intel's OpenCL compiler (IGC, compute-runtime 26.27) exposes what is missing:
`intel_sub_group_block_read*` (one message a tile), `intel_sub_group_f16_f16_matrix_mad_k16`
(DPAS on plain registers, so pre-packed VNNI operands go straight in),
`cl_intel_subgroup_local_block_io` (block SLM I/O) and `cl_intel_media_block_io`
(2D reads). Measured (Run 28, `src/bench/opencl_gemm8_blk.cl`, numerically identical to
the scalar kernel):

| shape | Vulkan staged | OpenCL block reads |
|---|---:|---:|
| 960x1536x512 | ~9.0 TF | 33.4 TF |
| 192x4096x1024 | ~8.7 TF | 32.4 TF |
| 4096x1024x4096 | ~10.0 TF | 25.7 TF |
| 9216x128x128 | ~7.0 TF | 15.4 TF |
| 36864x128x64 | ~7.0 TF | 9.7 TF (memory-bound, fp32 out) |

Run 23 found the OpenCL frame *slower* (201 ms, fusions off). That used the scalar-gather
kernel and none of the fused passes, so it does not predict this port. It does set the bar:
**each family is ported fused, and wins in the frame, or the port stops there.**

## 2. What exists

- `src/gpu/libopenclmx.c` (1309 lines): the same `xmx_*` C ABI as `libxmx.c`, so
  `xmxres.py`, `nr_frame_resident.py` and `src/ref/nr_frame.c` need no new calls.
  Working: buffer layer with phase65's discrete split (Run 21), general strided/batched/
  transposed GEMM, group-GEMM offsets (Run 22), unary, row, copy, `XMX_TIME` events (Run 23).
  Stubs (return 0 and do nothing): `rec_gemm_residual`, `_window_residual`(`_pool`),
  `_qkv`, `_qkv_window`, `_dual`, `rec_qkv`, `rec_window_attention`, `rec_ffn`(`_merge`,
  `_stem`), `rec_window_block`, `rec_global_attention`, `rec_history`, `rec_gemm_int8`,
  and every `xmx_profile_*`.
- Not wired: `NR_GPU_BACKEND=opencl` is refused by `src/nr_build.py` and
  `src/ref/nr_frame.c` (Runs 21-23 used an uncommitted selector); no Makefile target.
- The live path is safe: the Vulkan layer only copies frames to `nr_daemon.py`, a separate
  process, so the daemon can run on OpenCL with no layer change. Only `nr_frame_adopt_vulkan`
  (in-process device sharing) has no OpenCL equivalent; it keeps Vulkan.
- No `cl_khr_command_buffer` on this driver: a frame is ~410 `clEnqueueNDRangeKernel`
  calls on one in-order queue (ordering is implicit; no barriers needed).

## 3. Design

**One backend, same ABI.** `libopenclmx.c` stays the only new runtime. The graph code
does not learn about OpenCL; `xmx_rec_*` keeps recording into a command list, and
`xmx_submit` replays it as kernel enqueues. `xmx_graph_capture/run` cache the list with its
argument blocks resolved (buffer handles, offsets, grids), so a replayed frame is only
`clSetKernelArg` + enqueue.

**Kernel sources in files.** Kernels move out of C string literals into `src/gpu/opencl/*.cl`
(one per family, shared `publish.clh` / `cosine_tree.clh` mirroring the GLSL includes), built
once per process with `-cl-std=CL3.0` and the family's `-D` specialisations, and cached as
device binaries (`clGetProgramInfo(CL_PROGRAM_BINARIES)`) under `work/clcache/` keyed by
source hash + driver version, so the daemon does not pay IGC compile time on every launch.

**GEMM: one kernel family, epilogues as specialisations.** The block-read DPAS core
(subgroup 8, RMxRN atoms of 8x8x16) with every epilogue `libxmx.c:record_gemm` dispatches:
plain / half out, e4m3 publish, gate, residual with cosine, window residual (scatter
back to the image), window residual + 2x2 pool, QKV epilogue (normalise Q and K, scale,
publish) with and without the window gather, and the dual (half copy) store. The A-side
gathers (window gather, half image) become the address computation of the block read; a
non-contiguous A row is read with per-row `block_read` (8 lanes x 32 bits, still one message).
Tails: M % 8 and N % 8 via a masked last tile, never via padding the graph.

**Weights in VNNI.** A GEMM's B operand is pre-packed into fragment order
(`pack_vnni` in `opencl_gemm8.py`): one 256-byte `block_read8` an atom. Weights are uploaded
once, so the backend packs them at upload into a second device buffer keyed by
(buffer, offset, K, N, ldb) and drops the copy when the buffer is re-uploaded or destroyed.
A B that is an activation (attention's K and V inside the fused kernels, any non-weight B)
takes the row-major path: `intel_sub_group_block_read_us` per K row pair, or a media block
read where the extent allows. The rule is decided by buffer kind, not guessed per call.

**Fused passes: same arithmetic, new loads.** Each fused kernel is ported from its GLSL
with the operations in the same order (`cosine_tree`, the softmax transform, publish), so
the Vulkan tests stay the oracle. What changes is data movement: weights and tiles by block
read, K/V and attention tiles via `intel_sub_group_block_read/write` on local memory,
subgroup 8 or 16 chosen per kernel by measurement.

**Rounding.** DG2's Vulkan path cannot ask for round-to-nearest-even half conversion; OpenCL
can (`convert_half_rte`, `vstore_half_rte`). The port uses RTE everywhere and records the
difference against Vulkan in each family's test. Exact equality with Vulkan is therefore not
the gate. The gate is the reference.

## 4. Correctness gates (per family, before its speed counts)

1. Its existing test with `NR_GPU_BACKEND=opencl`: `test_gemm_contract`, `test_gemm_residual`,
   `test_gemm_qkv`, `test_qkv_fusion`, `test_ffn_fused`, `test_ffn_batch`,
   `test_window_attention`, `test_window_block`, `test_global_attention`, `test_resident`,
   `test_frame_execution`. Each must pass, or the deviation must be explained by RTE alone
   (shown by the same test against the CPU reference).
2. Whole frame: `test_resident` frame correlation against the host reference at least the
   Vulkan path's value; head hash recorded; `src/ref/test_nr_frame_c.py` against a reference
   recorded on OpenCL (`work/nr_frame_reference_opencl.bin`), as Metal has its own.
3. Real frames: `nr_frame --replay` over the Ghost moving segment (frames 350-365,
   `work/video/ghost-full-rs05/in/`): PSNR and MAE against the Vulkan output, plus the owner's
   look. Graphical verdicts are the owner's (AGENTS.md).

## 5. Phases

Each phase ends with: the tests above for what it touched, `frame_profile` on both
backends, a diary entry with error counts, and a commit. A phase that loses in the frame
is recorded and either fixed or stopped. It is never merged on a microbenchmark.

**P0 - plumbing (no speed yet).**
`NR_GPU_BACKEND=opencl` in `nr_build.py`, `xmx.py`, `nr_frame.c`; Makefile target
`work/libopenclmx.so` (gcc-13, `libOpenCL.so.1`, no CL headers needed: the declarations stay
local as now); `src/gpu/opencl/` source loading and the binary cache; `xmx_profile_*` from
`cl_event` timestamps with the same kind numbers as libxmx, so `frame_profile.py`,
`iterate.sh` and `runs.jsonl` work unchanged; `iterate.sh` takes `NR_GPU_BACKEND` from `ENV`.
Exit: the Vulkan-only parts of `make test` still pass, and the OpenCL backend runs the
unfused frame (Run 23's 201 ms) with a per-pass table.

**P1 - GEMM (50 % of the frame).**
Block-read kernel with strides, offsets, batch, transpose, tails; then the epilogues in
order of frame time: plain + publish (`gemm 2304x128x256 x8`, `9216x128x128 x4`, ...),
residual, QKV + window gather, window residual (+ pool), dual. VNNI weight packing at upload.
Per-shape tile choice (RMxRN, subgroup) from a sweep over the 68 frame shapes
(`src/bench/census.py` shapes, `replay.py`), table compiled into the backend.
Also: why tiles above 8 atoms collapse (IGC dump: `IGC_ShaderDumpEnable=1`,
`IGC_DumpToCustomDir`), since a 32x32 tile is the textbook DG2 optimum.
Exit: frame GEMM time under Vulkan's 41.4 ms by at least 2x (aim 15-20 ms), gates 1-2.

**P2 - elementwise and history.** `rec_history`, the unary/row kinds the fused frame uses,
`rec_qkv`. Small (0.7 ms), but needed for a whole fused frame.

**P3 - fused attention and FFN, in order of frame time.**
1. `window_block` (22.5 ms): one 8x8 window a workgroup today, 16 KB SLM, 8 KB of weights
   re-read per window, 75 spills. Port as is first, then: several windows a workgroup so the
   QKV/projection weights are loaded once into SLM by block read; Q, K, V and logits in DPAS
   registers at subgroup 8 (an 8x8 fp32 accumulator is one register a lane there, not 4).
2. `ffn_fused` + merge + stem (8.5 ms).
3. `window_attention` (8.0 ms, 52 passes).
4. `global_attention` (1.1 ms).
Exit per kernel: faster than its Vulkan pass in the frame, gate 1. Whole-family exit: gates 2-3.

**P4 - the live path.** `nr_daemon.py` and `nr_frame --replay`/`--stream` on
`NR_GPU_BACKEND=opencl`; first-launch compile time measured with the binary cache cold and
warm; the owner plays it through `18-launch-user.sh` with the layer.

**P5 - past one GPU, if P1-P3 land short.** The two-GPU alternate-frame feeder (Run 25:
two devices do not contend; stale history measured as a legitimate render) and/or
`NR_DEEP_CACHE`. Both are orthogonal to the backend and stack on it.

## 6. Budget (what has to be true for 24-30 fps)

| family | Vulkan | target | basis |
|---|---:|---:|---|
| GEMM | 41.4 | 15-20 | 2-3.7x measured on frame shapes; small-N shapes nearer 1.5-2x |
| window block | 22.5 | 8-11 | weights loaded once a workgroup, no spills, DPAS at SG8 |
| ffn fused | 8.5 | 4-5 | same GEMM core inside |
| window attention | 8.0 | 4-5 | block SLM I/O, SG8 accumulators |
| global + unary + history | 1.8 | 1.8 | |
| host / enqueue | (9.9 wall) | <= 5 | 410 enqueues; overlap with device via in-order async queue |
| **device total** | **82.3** | **33-43** | |

The upper end of the range meets the target with one GPU. If it lands at 43-50 ms, P5's
second GPU covers the rest.

## 7. Risks

- **Small-N shapes** (N = 32-128, M up to 590 k) are half the GEMM passes and gain least
  from block reads. They are bandwidth- and occupancy-bound; the lever is fusion (the
  epilogue writes half and publishes in place) and wider M tiles, not DPAS rate.
- **Enqueue overhead without command buffers**: ~410 enqueues a frame at a few us each is
  1-4 ms of host time. That's acceptable if it overlaps device execution. Measure in P0.
- **IGC register allocation cliff** (Run 28: >8 atoms collapses with no reported spill):
  understand it in P1 or the tile sizes stay small.
- **Numerics**: RTE versus DG2-Vulkan's driver-default rounding; e4m3 publish emulation must
  match `publish.glsl` bit for bit on its own inputs (unit test on all 65 536 halves).
- **Two runtimes in the tree**: Vulkan stays the default and the reference for the
  portable/Xe2/Metal/D3D12 paths; OpenCL is opt-in until P4 passes, then the DG2 default.

## 8. Not doing

Patching Mesa's cooperative-matrix lowering (it would fix every Vulkan kernel at once, but it
means carrying a driver build), SYCL/oneAPI (a compiler toolchain the box does not have, for
the same IGC underneath), and Level Zero (no gain over OpenCL for these kernels on DG2).
