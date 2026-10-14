# Porting the frame to OpenCL (Intel DG2 / Arc A-series)

Status: **the backend exists and runs; this is the remaining-work plan** (rev 2026-10-14,
after Run 45). The OpenCL frame is live at **95.6 ms device** (1024x576, GPU1) against
Vulkan's **82.3 ms**. First milestone is to beat Vulkan; then 24-30 fps (33-41 ms) on one
A770. Every number here is measured in the frame (`frame_profile`), never a microbenchmark.

## 1. Where the frame time is

OpenCL at 1024x576 on GPU1 (`frame_profile --calls`, Run 45); Vulkan's column is Run 26:

| pass family | OpenCL ms | Vulkan ms | passes | us each |
|---|---:|---:|---:|---:|
| GEMM (all epilogues, 8x16 staged) | 60.7 | 41.4 | 339 | 179 |
| row: window attention | 15.6 | 8.0 | 62 | 251 |
| ffn fused (+ merge, stem) | 14.3 | 8.5 | 10 | 1425 |
| row: global attention | 1.9 | 1.1 | 8 | 236 |
| unary (partition, pool2, upsample, e4m3, pad) | 3.3 | 0.7 | 83 | |
| **device total** | **95.6** | **82.3** | 499 | |
| host overhead (recording, numpy, the rest) | 20.9 wall | | | |

Two measurements now decide the plan, and both close the original premise of the port:

- **DPAS is closed (Run 45).** The block-read kernel benches at 16.5 TF/s on GPU1, yet
  in the frame the block path takes 1617/2149 calls and moves device time **0 %** (94.77 vs
  the FMA baseline 95.75); the 16x16 tile is worse (104.79). The GEMM family is **196
  GFLOP/frame at ~3.35 TFLOP/s, arithmetic intensity ~31** - under even the 7.4 TF/s FMA
  ceiling and nowhere near 512 GB/s. It is not multiply-bound, so XMX cannot help it.
- **The GEMM is flat across ~25 shapes at 2-4 ms each.** The huge-M skinny ones (589824 or
  147456 x 32/96 x 32/128) run at **1.3-1.6 TF/s**: M/(8*rm) = 36864+ workgroups of 8
  threads. The cost is workgroup count / occupancy and the per-element publish epilogue, not
  the atom.

The reason OpenCL is still the right runtime stands where it matters - IGC exposes the
subgroup block-IO and DPAS extensions Mesa cannot reach (`VK_INTEL_shader_integer_functions2`
is all ANV offers) - but those are not the GEMM lever. The port is accepted on frame speed.

## 2. What exists

- `src/gpu/libopenclmx.c` (~2900 lines): the same `xmx_*` C ABI as `libxmx.c`, so
  `xmxrt`, `nr_frame_resident.py` and `src/ref/nr_frame.c` need no new calls. The whole
  fused frame runs on it: general strided/batched/transposed GEMM, the fused QKV-epilogue
  projection (Run 43), the fused on-chip window attention (Run 42), the fused FFN, unary,
  row, copy, `XMX_TIME` events, `xmx_graph_capture/run`, and `xmx_profile_*` with the same
  kind numbers as libxmx. Device selection is `XMX_DEVICE` (GPU index; two A770 devices
  share one platform).
- The default GEMM is the register-blocked FMA core (Run 24/43). The block-read DPAS kernel
  is present behind `XMX_CL_BLK` and off (Run 45).
- `NR_GPU_BACKEND=opencl` is wired in `src/nr_build.py`; the default stays Vulkan.

## 3. Design

**One backend, same ABI.** `libopenclmx.c` stays the only new runtime. The graph code
does not learn about OpenCL; `xmx_rec_*` keeps recording into a command list, and
`xmx_submit` replays it as kernel enqueues. `xmx_graph_capture/run` cache the list with its
argument blocks resolved (buffer handles, offsets, grids), so a replayed frame is only
`clSetKernelArg` + enqueue.

**Kernels are C string literals today** (`GEMM_SRC`, `ATTN_SRC`, `RESIDENT_SRC`, ... in
`libopenclmx.c`), built once per process with the family's `-D` specialisations. Moving them
to `src/gpu/opencl/*.cl` with a `work/clcache/` device-binary cache (keyed by source hash +
driver version) so the daemon does not pay IGC compile time on every launch is P4, not yet
done. `CL_COMMON` sets `#pragma OPENCL FP_CONTRACT OFF`, which the publish epilogue's
rounding depends on.

**GEMM: one kernel family, epilogues as specialisations.** The register-blocked FMA core
with every epilogue `libxmx.c:record_gemm` dispatches, which is what ships: plain / half out,
e4m3 publish, gate, residual with cosine, window residual (scatter back to the image),
window residual + 2x2 pool, QKV epilogue (normalise Q and K, scale, publish) with and without
the window gather, and the dual (half copy) store. Subgroup 8, RMxRN atoms of 8x8x16. Tails:
M % 8 and N % 8 via a masked last tile, never via padding the graph. A block-read DPAS
variant of the same kernel exists behind `XMX_CL_BLK`; it is off because it does not move the
frame (Run 45). Raising the atom above 2 atoms is the open question in P1.

**Weights in VNNI (block-read path only).** When `XMX_CL_BLK` is on, a GEMM's B operand is
pre-packed into fragment order (`pack_vnni` in `opencl_gemm8.py`): one 256-byte `block_read8`
an atom, cached per buffer and generation in `ensure_packed`. Off, B is read row-major
directly by the FMA core.

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

## 5. Remaining work

Each item ends with: the tests above for what it touched, `frame_profile` on OpenCL (and
Vulkan where it moved), a diary entry with error counts, and a commit. An item that does not
move the frame is recorded and either fixed or dropped. It is never merged on a
microbenchmark.

**P1 - the GEMM (60.7 ms, 339 passes, 179 us each).** Two measured levers, in order:

1. **Occupancy on the huge-M skinny shapes** (589824/147456 x 32/96 x 32/128, 1.3-1.6 TF/s).
   Today a workgroup is 8 threads - one subgroup, RMxRN = 2x1 - and a 589824-row shape makes
   36864+ of them. Raise threads a workgroup: **several N-subgroups that share one loaded A
   tile** (RN subgroups, each an 8-column strip), and/or an M tile above 2 atoms. This is not
   the private-array 4x4 tile Run 39 measured (that was slower); here the extra work is other
   subgroups reusing A, not a bigger per-thread register file.
2. **The publish epilogue** (~35 ms of the 62, Run 43; half of it `HROUND`). Publish in place
   from the accumulator where the store allows, instead of a half round-trip.
3. Leave the block-read DPAS kernel off (`XMX_CL_BLK`) until a shape actually waits on the
   multiply (Run 45).

Exit: GEMM under Vulkan's 41.4 ms, gates 1-2.

**P2 - host overhead (20.9 ms wall over device).** 499 enqueues a frame plus numpy recording.
Overlap with the device on the in-order async queue and measure enqueue cost in the frame;
1-4 ms of host time is fine only if it hides behind the GPU.

**P3 - the fused passes, by frame time.**

1. `window attention` (15.6 ms, 62 passes, 251 us each): the per-token rows and the
   split/merge schedules; the on-chip fused version already replaced the decomposed passes
   (Run 42).
2. `ffn fused` (14.3 ms, 10 passes, 1425 us each): the 589824x32x128 merge and stem dominate
   at 1.3 TF/s - the same occupancy problem as P1, inside the fused kernel.
3. `global attention` (1.9 ms).

Exit per kernel: faster than its Vulkan pass in the frame; gate 1.

**P4 - the live path.** `nr_daemon.py` and `nr_frame --replay`/`--stream` on
`NR_GPU_BACKEND=opencl`; the binary cache keeps IGC's first-launch compile off the critical
path; the owner plays it through `18-launch-user.sh`.

**P5 - past one GPU.** The two-GPU alternate-frame feeder (Run 25: two devices do not
contend; stale history measured as a legitimate render) and/or `NR_DEEP_CACHE`. Both are
orthogonal to the backend and stack on it.

## 6. Budget (what has to be true for 24-30 fps)

| family | Vulkan | OpenCL now | target | basis |
|---|---:|---:|---:|---|
| GEMM | 41.4 | 60.7 | 15-20 | not multiply-bound; workgroup count + epilogue (Run 45) |
| window attention | 8.0 | 15.6 | 4-5 | 62 passes at 251 us |
| ffn fused | 8.5 | 14.3 | 4-5 | occupancy inside the kernel |
| global + unary + history | 1.8 | 5.0 | 1.8 | |
| host / enqueue | (9.9 wall) | 20.9 | <= 5 | 499 enqueues; overlap with device via in-order async queue |
| **device total** | **82.3** | **95.6** | **33-43** | |

The upper end of the range meets the target with one GPU. If it lands at 43-50 ms, P5's
second GPU covers the rest.

## 7. Risks

- **Small-N shapes** (N = 32-128, M up to 590 k) are half the GEMM passes and run at
  1.3-1.6 TF/s. The lever is occupancy (more subgroups a workgroup) and publishing the
  epilogue in place, not DPAS rate (Run 45).
- **Enqueue overhead without command buffers**: 499 enqueues a frame at a few us each is
  1-4 ms of host time, on top of 20.9 ms of recording/numpy. Only free if it overlaps the
  device. This is P2.
- **Numerics**: RTE versus DG2-Vulkan's driver-default rounding; e4m3 publish emulation must
  match `publish.glsl` bit for bit on its own inputs (unit test on all 65 536 halves). The
  suite's cross-kernel comparisons are bit-exact, so the OpenCL paths must agree with each
  other exactly even though they need not equal Vulkan.
- **Two runtimes in the tree**: Vulkan stays the default and the reference for the
  portable/Xe2/Metal/D3D12 paths; OpenCL is opt-in until P4 passes, then the DG2 default.

## 8. Not doing

Patching Mesa's cooperative-matrix lowering (it would fix every Vulkan kernel at once, but it
means carrying a driver build), SYCL/oneAPI (a compiler toolchain the box does not have, for
the same IGC underneath), and Level Zero (no gain over OpenCL for these kernels on DG2).
