# xmxrt - architecture of the resident GPU runtime (Python side)

Status: **proposal, not implemented.** This document describes the split of
`src/gpu/xmxres.py` into the `src/gpu/xmxrt/` package and the migration of every
consumer. Nothing here is built yet; section 11 is the order of work and section 12
the open decisions. Line numbers refer to `xmxres.py` at 1533 lines (working tree on
top of `1251d59`).

---

## 1. Scope

`xmxres.py` is the Python half of the resident runtime: it binds the native
`xmx_*` library (Vulkan `libxmx.c`, OpenCL `libopenclmx.c`, Direct3D 12
`libd3dmx.c`, Metal `libmetalmx.m`, all behind the one ABI in `xmx.h`), owns
device buffers, records passes into a command graph, and submits it. Everything
above it - the model graph in `nr_resident.py`, the frame driver in
`nr_frame_resident.py`, `src/ref/nr_frame.py`, the daemon, 34 tests, ~40 benches -
talks to the GPU through it.

Out of scope:

- The C ABI (`xmx.h`) and every `xmx_*` backend. The split is Python-internal;
  no native symbol, argument order or flag bit changes.
- `src/ref/nr_frame.c`, the C mirror of the frame. It mirrors `Runtime.__init__`
  switch for switch; only its comments name `xmxres.Runtime` and they are
  updated, nothing else.
- `xmx.py` (the older single-pass binding, 364 lines). It is a separate module
  and stays separate.

## 2. Current state

### 2.1 Size

| File | Lines |
|---|---|
| `src/gpu/xmxres.py` | 1533 |
| all other non-test Python in `src/gpu` | 2398 |
| tests in `src/gpu` | 4515 |

`xmxres.py` is 39 % of the non-test GPU Python. It lives under a grandfathered
override in `config/file_length_limits.yaml` (`max_lines: 1527`; default 512),
and it is already 6 lines over that override, which is why the Vulkan
window-block default (Run 60) cannot be committed.

### 2.2 What is in the file

| Lines | Concern | Symbols |
|---|---|---|
| 1-35 | module doc, imports, repo root, `nr_build` | `ROOT` (repo-root computation), the `nr_build` import (import-order suppression) |
| 36-58 | op kinds and flag builders | `E4M3 ... POOL2_SKIP`, `EPI_*`, `TM/TN/TK`, `_reads`, `_publish` |
| 60-193 | native library | `_lib`, `DeviceLost`, `failure`, `_load` (ctypes `argtypes`/`restype` via `getattr`, the `.spv` inventory) |
| 196-211 | helpers | `fused_shader`, `align`, `GRAPH`, `HOST_READ`, `HOST_WRITE` |
| 214-344 | memory | `Buffer`, `host_view`, `host_write`, `CommandGraph` |
| 347-529 | scratch reuse | `ScratchArena`, `_ScratchBuffer`, `_ScratchHandle` |
| 532-578 | profiling | `PROFILE_FAMILIES`, `profile`, `profile_reset`, `profile_each`, `profile_each_kinds`, `profile_totals` |
| 581-1520 | `Runtime` | see 2.3 |
| 1521-1533 | `_Independent` | context manager returned by `Runtime.independent()` |

### 2.3 What is in `Runtime` (~940 lines, ~75 methods)

1. **Configuration** (584-695). 21 boolean switches read from the environment
   in `__init__` (`fuse_qk`, `batch_ffn`, `input_fp16`, `compact_head`,
   `joint_qkv`, `fuse_residual`, `fuse_window_residual`,
   `fuse_window_attention`, `fuse_attention_merge`, `qkv_epilogue`,
   `fuse_glue`, `fuse_ffn`, `fuse_branched_ffn`, `fuse_partition`,
   `fuse_transition`, `fuse_merge_ffn`, `fuse_stem_ffn`, `fuse_pool`,
   `fuse_window_block`, `fuse_head`, `fuse_global_attention`), two of them also
   gated on the device (`xmx_window_gather()`), one on the backend
   (`fuse_window_block`). Each carries its measurement history as a comment.
   `graph_key()` packs them with `xmx_specialization()` into the graph-replay
   cache key; `scratch_key()` masks off the low three bits.
2. **Allocation** (696-735): `buffer_bytes`, `staging`, `buffer`, `read`,
   `write`, `buffer_from`.
3. **Recording** (736-798, 1514): `begin`, `prepare_qkv`, `independent`,
   `capture`, `abort`, `copy`, `specialize`, `submit`.
4. **Passes** (799-1513), by family:
   - GEMM: `gemm`, `gemm_residual`, `gemm_residual_pool`, `gemm_qkv`,
     `gemm_dual`, `gemm_int8`, `ffn_fused`, `ffn_fused_merge`,
     `ffn_fused_stem`.
   - Attention: `window_block`, `window_attention`, `global_attention`,
     `softmax`, `vit_softmax`, `cosine_publish`, `add_bias`.
   - Element-wise: `unary` and its wrappers (`e4m3`, `gate`, `half`,
     `gate_e4m3_half`, `e4m3_half`, `gate_half`, `to_half`, `from_half`,
     `scale`), `residual`, `add`, `scale_channel`, `upsample_merge`,
     `upsample2`, `upsample_add`, `pool2`, `pool2_skip`, `pad_end`,
     `sample_history`.
   - Layout: `window_extent` (static), `partition`, `reverse`, `split_heads`,
     `merge_heads`.

Every pass method does the same three things: validate shapes and arguments,
build a flag word (`_reads`, `_publish`, `EPI_*`), call one `self.lib.xmx_rec_*`
and bump `self.recorded`. The only state a pass touches is `self.lib`,
`self.recorded` and the switches.

### 2.4 Problems

1. **Six unrelated concerns in one module.** Loading a shared library, a
   value-object of feature switches, a buffer type, a scratch allocator, a
   profiler and ~60 pass encoders change for different reasons and at different
   rates, yet every change touches the same file and its hooks.
2. **Configuration is mutable state on the device object.** Tests flip
   `rt.fuse_window_block = False` on a live runtime; nothing re-derives the
   graph key, nothing stops a switch changing between capture and replay. The
   key is correct only because callers happen to set switches before recording.
3. **The configuration has no single owner.** The environment is read in
   `Runtime.__init__`; the same decisions are re-made in `nr_frame.c`; the
   backend-dependent default for `fuse_window_block` lives in a constructor.
4. **Pass encoders are welded to the device.** They need only "a library and a
   counter", but they can only be reached through a fully initialised device,
   so tests use `object.__new__(xmxres.Runtime)` (`test_ffn_batch.py:101`) and
   benches monkeypatch `xmxres.Runtime.gemm` (`census.py`, `census_ew.py`,
   `int_render.py`, `int_weights.py`).
5. **Hooks cannot be satisfied honestly.** The file is over its grandfathered
   limit; every further change means editing the override or shaving unrelated
   code. Three banned-word exemptions are scoped to the whole file although the
   constructs they cover live in 30 lines of it.

## 3. Goals and non-goals

Goals:

- One module per concern, every module under the 512-line default; the
  `xmxres.py` override is removed, not raised.
- Configuration is an immutable value with one constructor and one key
  function; the graph key is bit-for-bit what it is today.
- Pass encoders are plain functions of an explicit recorder; testable without
  a device.
- No behaviour change: the recorded pass stream, graph keys and frame bytes are
  identical before and after, on every backend we can run (Vulkan, OpenCL).
- Every measurement comment on a switch survives verbatim.

Non-goals:

- No new switches, no default changes inside the refactor (the Vulkan
  window-block default is its own commit, section 11).
- No performance work. A refactor commit that moves timing outside noise is
  investigated, not accepted.
- Nothing kept only to re-export the old names: `xmxres.py` is deleted at the end.

## 4. Target layout

```
src/gpu/xmxrt/
    __init__.py      docstring only; no re-exports
    native.py        the native library
    ops.py           op kinds, epilogue codes, flag builders
    memory.py        device buffers and command graphs
    scratch.py       scratch arena
    profile.py       per-pass profiling
    config.py        FuseConfig
    recorder.py      Recorder
    gemm.py          GEMM-family pass encoders
    attention.py     attention-family pass encoders
    elementwise.py   element-wise and layout pass encoders
```

Why a subpackage: `src/gpu` already holds 44 direct source files against the
module-size limit of 56, and two child directories (`d3d12`, `metal`) against
8. Ten new flat files would leave one slot; a package costs one child slot and
groups the runtime visibly.

`__init__.py` re-exports nothing. A barrel would recreate the monolith's
coupling under a new name; importers name the module that owns the symbol.

### 4.1 Dependency graph

```
                native            (ctypes, nr_build, ROOT)
               /   |    \
            ops  memory  profile
               \   |
               config            (native: device capability queries)
                   |
               recorder ---- scratch
              /    |    \
          gemm attention elementwise   (ops, memory, recorder)
```

Rules, enforced by review and by an import test (section 10.4):

- Arrows point down only. No module imports a sibling above it.
- Pass modules never import each other.
- Only `native.py` imports `ctypes`, reads `ROOT`, or imports `nr_build`.
- Only `config.py` reads `os.environ` for switches.

## 5. Module contracts

Sizes are estimates from the current line ranges.

### 5.1 `native.py` (~200)

Owns the shared library and nothing else.

```
ROOT: Path
class DeviceLost(RuntimeError): ...
def failure(lib, call) -> RuntimeError        # DeviceLost when xmx_device_lost()
def load() -> ctypes.CDLL                      # was _load(); cached, signatures bound
def fused_shader(lib, name) -> bool
SHADERS: tuple[str, ...]                        # the .spv inventory, one place
```

- Contains the three constructs the banned-word exemptions cover (`getattr`
  with a literal name for `argtypes`/`restype`, the repo-root computation, the
  import-order suppression). The exemptions move here and are no longer
  whole-file grants on 1500 lines.
- `SHADERS` is the literal list the tooling scans for `.spv` names
  (`build_check.py`, `wizard/process.py` (`_shader_names`),
  `scripts/build_release.py`). Keeping every literal here means one path for
  them to scan.

### 5.2 `ops.py` (~50)

Pure constants and the two flag builders; no imports beyond the standard
library.

```
E4M3, GATE, HALF, ..., POOL2_SKIP           # unary/op kinds
COSINE_PUBLISH, SOFTMAX, VIT_SOFTMAX
EPI_NONE, EPI_GATE, EPI_E4M3, EPI_GATE_E4M3, ...
TM, TN, TK
GRAPH, HOST_READ, HOST_WRITE
def reads(a_half=False, b_half=False) -> int   # was _reads
def publish(epilogue, narrow) -> int           # was _publish
```

The values are part of the ABI with the C side (`libopenclmx.c:1902` cites
`GRAPH/HOST_*`); they move unchanged.

### 5.3 `memory.py` (~150)

```
class Buffer           # nbytes, kind; mapped, view, upload, download, zero, free
def host_view(buffer, dtype=np.float32, shape=None, count=None)
def host_write(buffer, array, rows=None)
class CommandGraph     # run, free
def align(value, multiple)
```

`Buffer` keeps obtaining the library through `native.load()` as it does today
through `_lib()`; no caller has to pass it.

### 5.4 `scratch.py` (~190)

`ScratchArena`, `_ScratchBuffer`, `_ScratchHandle`, unchanged in behaviour.
The arena today calls `runtime.buffer(...)`; it will take any object with that
method (the `Recorder`), so it does not import `recorder.py` and the arrow
stays one-way.

### 5.5 `profile.py` (~50)

`PROFILE_FAMILIES`, `profile`, `profile_reset`, `profile_each`,
`profile_each_kinds`, `profile_totals`. Thin wrappers over `native.load()`.

### 5.6 `config.py` (~150) - `FuseConfig`

```
_FIELDS = ("fuse_qk", "batch_ffn", "input_fp16", "compact_head", "joint_qkv",
           "fuse_residual", "fuse_window_residual", "fuse_window_attention",
           "fuse_attention_merge", "qkv_epilogue", "fuse_glue", "fuse_ffn",
           "fuse_branched_ffn", "fuse_partition", "fuse_transition", "fuse_merge_ffn",
           "fuse_stem_ffn", "fuse_pool", "fuse_window_block", "fuse_head",
           "fuse_global_attention")

class FuseConfig(namedtuple("FuseConfig", _FIELDS)):
    __slots__ = ()
    @classmethod
    def from_environment(cls, lib, backend=None): ...    # one switch per field
    def with_config(self, **changes): ...                # a new value object
    def key(self, specialization): ...                   # was graph_key
    def scratch_key(self, specialization): ...
```

- **Field order, defaults, environment names and comments move verbatim.**
  Each field keeps the comment block that today sits above its line in
  `__init__` (sources: 589-664). The measurement history is the reason a
  default is what it is; it is not trimmed.
- `from_environment` is the only reader of `NR_*` switch variables. It takes
  the library for the two capability gates (`xmx_window_gather()`) and the
  backend (`nr_build.backend()` by default) for `fuse_window_block`.
- `key()` reproduces lines 667-688 bit for bit, including the comment on why
  bits 5-7 are not ProjectsCodex's. The bit table is fixed:

  | bit | source | bit | source |
  |---|---|---|---|
  | 0-2 | `xmx_specialization()` | 13 | `fuse_glue` |
  | 3 | `fuse_qk` | 14 | `fuse_ffn` |
  | 4 | `batch_ffn` | 15 | `fuse_branched_ffn` |
  | 5 | `input_fp16` | 16 | `fuse_partition` |
  | 6 | `compact_head` | 17 | `fuse_transition` |
  | 7 | `joint_qkv` | 18 | `fuse_merge_ffn` |
  | 8 | `fuse_residual` | 19 | `fuse_stem_ffn` |
  | 9 | `fuse_window_residual` | 20 | `fuse_pool` |
  | 10 | `fuse_window_attention` | 21 | `fuse_window_block` |
  | 11 | `fuse_attention_merge` | 22 | `fuse_head` |
  | 12 | `qkv_epilogue` | 23 | `fuse_global_attention` |

  The key is written out as the explicit `|`/`<<` expression it is today, not
  generated from a table: it is the replay cache key, a wrong bit replays the
  wrong graph with no error anywhere, and the explicit form is what reviewers
  check.
- Frozen: a recorder's configuration cannot change after it is built, which
  closes problem 2.4.2. Variants are made with `FuseConfig.with_config`.

### 5.7 `recorder.py` (~250) - `Recorder`

The object that replaces `Runtime`. It owns what is genuinely per-device and
per-recording, and nothing else.

```
class Recorder:
    def __init__(self, config: FuseConfig | None = None, lib=None): ...
    lib                     # the bound native library
    config: FuseConfig
    recorded: int           # passes recorded since begin()

    # identity
    def graph_key(self) -> int        # config.key(lib.xmx_specialization())
    def scratch_key(self) -> int

    # allocation
    buffer_bytes, staging              # properties
    def buffer(self, count, dtype=np.float32, kind=GRAPH)
    def read(self, buffer, ...)        # host_view, for src/ref which holds an object
    def write(self, buffer, ...)       # host_write, same reason
    def buffer_from(self, array, dtype=np.float32, pad=0)

    # recording
    def begin(self)
    def note(self, count=1)            # pass encoders call this instead of += on .recorded
    def prepare_qkv(self, ...)
    def independent(self) -> _Independent
    def capture(self) -> CommandGraph
    def abort(self)
    def copy(self, ...)
    def specialize(self, mask=7)
    def submit(self)
```

- `config` is `FuseConfig.from_environment(lib)` when a caller passes none, so
  `Recorder()` behaves exactly like `Runtime()`.
- `lib` is injectable: a test can hand a recording stand-in and a config and
  exercise encoders with no device (replaces `object.__new__`).
- `_Independent` moves here; it is the recorder's own context manager.

### 5.8 Pass modules

Each pass is a function whose first argument is the recorder:

```
# gemm.py
def gemm(rec, a, b, c, rows, cols, inner, *, batch=1, strides=None,
         transpose_b=False, ...): ...
```

The body is the current method body with `self` renamed `rec` and
`self.recorded += 1` replaced by `rec.note()`. Docstrings and comments move
with the function. Validation, flag construction and the `xmx_rec_*` call are
unchanged.

| Module | Functions | ~Lines |
|---|---|---|
| `gemm.py` | `gemm`, `gemm_residual`, `gemm_residual_pool`, `gemm_qkv`, `gemm_dual`, `gemm_int8`, `ffn_fused`, `ffn_fused_merge`, `ffn_fused_stem` | ~400 |
| `attention.py` | `window_block`, `window_attention`, `global_attention`, `softmax`, `vit_softmax`, `cosine_publish`, `add_bias` | ~300 |
| `elementwise.py` | `unary` + wrappers, `residual`, `add`, `scale_channel`, `upsample_merge`, `upsample2`, `upsample_add`, `pool2`, `pool2_skip`, `pad_end`, `sample_history`, `window_extent`, `partition`, `reverse`, `split_heads`, `merge_heads` | ~350 |

If `elementwise.py` lands above ~450 lines the layout passes go to their own
`layout.py`; the boundary is already clean (they share no helper with the
element-wise passes).

## 6. Data flow of a frame

Unchanged by the split; restated so the module boundaries can be checked
against it.

1. `nr_frame_resident.ResidentFrame` builds a `Recorder` (config from the
   environment) and a `ScratchArena` keyed by `rec.scratch_key()`.
2. Weights go up once through `rec.buffer_from` (`memory.Buffer`).
3. Per frame, in replay mode (`NR_FRAME_MODE=replay`, the default): look up a
   captured `CommandGraph` by `rec.graph_key()`; if present, `run()` it.
4. Otherwise `rec.begin()`, then `nr_resident` walks the model and calls pass
   encoders (`gemm.gemm(rec, ...)`, `attention.window_block(rec, ...)`, ...),
   choosing fused or decomposed paths from `rec.config`; each encoder calls one
   `rec.lib.xmx_rec_*` and `rec.note()`.
5. `rec.capture()` returns the `CommandGraph` stored under the key, or
   `rec.submit()` runs a single-shot recording (`NR_FRAME_MODE=single`, what
   `frame_profile.py` forces).
6. Host reads go through `memory.host_view`.

Configuration enters at exactly one point (1), is immutable afterwards, and is
the only input to the key besides the device specialisation.

## 7. Consumer migration

### 7.1 Inventory

`xmxres` is referenced from 76 files besides itself (64 Python, the rest C
sources, docs and notes). By kind:

| Consumer | Uses | Change |
|---|---|---|
| `src/gpu/nr_resident.py` (760) | ~40 pass methods, 16 `runtime.fuse_*`/switch reads, `begin/independent/prepare_qkv/submit/buffer/buffer_from` | passes become module calls; switch reads become `runtime.config.*` |
| `src/gpu/nr_frame_resident.py` (648) | `begin/capture/abort/copy/submit/staging/buffer/graph_key/scratch_key`, ~15 passes, `fuse_glue/fuse_head/compact_head/input_fp16` | same |
| `src/ref/nr_frame.py:460` | `xmxres.Runtime()` | `recorder.Recorder()` |
| `src/layer/nr_daemon.py:57,949` | `DeviceLost`, `Runtime()` probe | `native.DeviceLost`, `Recorder()` |
| `src/layer/test_daemon.py:263-273` | `DeviceLost` identity, `failure` | `native.*` |
| 34 tests in `src/gpu` | `Runtime()`, passes, switch writes, `host_view/host_write`, `EPI_*` | see 7.3 |
| ~40 benches in `src/bench` | `Runtime()`, passes, profiling, monkeypatches | see 7.4 |
| `src/tools/build_check.py:39`, `src/tools/wizard/process.py` (`_shader_names`), `scripts/build_release.py:41` | scan `xmxres.py` for `.spv` names | scan `xmxrt/native.py` |
| `src/tools/wizard/config.py` (`RUNTIME_MODULES`) | ships `xmxres.py` to Windows | ship the 11 package files |
| `src/tools/test_release.py:29` | copies `xmxres.py` into a fake tree | copy the package |
| comments: `libopenclmx.c:1733,1902,2877,2939`, `libd3dmx.c:5`, `nr_frame.c:1546,1738`, `docs/opencl-port.md`, `docs/WINDOWS-PORT.md` | name `xmxres.*` | renamed to the new owner |

`notes/phase*.md` and `notes/AUDIT-*.md` are history and are left as written.

### 7.2 Graph code

Mechanical, one module at a time:

```
# before
runtime.gemm(a, b, c, rows, cols, inner, transpose_b=True)
if runtime.fuse_window_block: ...
# after
gemm.gemm(runtime, a, b, c, rows, cols, inner, transpose_b=True)
if runtime.config.fuse_window_block: ...
```

The variable keeps its name (`runtime`, `rt`) to keep the diff reviewable;
only the call shape changes.

### 7.3 Tests

- Switch writes on a live object (`rt.fuse_window_block = False`) become
  `rt.with_config(fuse_window_block=False)`. Tests that compare fused against
  decomposed run each path on its own config.
- `object.__new__(xmxres.Runtime)` (`test_ffn_batch.py:101`) becomes a
  recorder-shaped stand-in carrying a config.
- `xmxres.EPI_*` becomes `ops.EPI_*`; `xmxres.host_*` becomes `memory.host_*`.
- No test changes what it asserts. A test whose assertion has to change is a
  behaviour change and stops the step.

### 7.4 Benches

- `xmxres.Runtime.gemm = wrapper` (census, census_ew, int_render,
  int_weights) becomes `gemm.gemm = wrapper` - patching the module attribute,
  which `nr_resident` resolves at call time because it calls `gemm.gemm(...)`
  through the module, not a bound name.
- `xmxres.profile*` becomes `profile.*`.

## 8. Tooling and configuration

Code (changed in the same commit as the move that needs it):

- `.spv` scanners (`build_check.py`, `wizard.process._shader_names`,
  `scripts/build_release.py`) read `src/gpu/xmxrt/native.py`.
- `RUNTIME_MODULES` lists every `src/gpu/xmxrt/*.py`.
- `test_release.py` copies the package.

Configuration (owner-only; supplied as `workspace-yaml-edit` commands, not
edited by an agent):

- `config/file_length_limits.yaml`: remove the `src/gpu/xmxres.py` override.
- `config/banned_words_exceptions_v5.yaml`: re-scope the three exemptions
  (lines 87 `bgetattr`, 615 `parents`, 1281 `s-noqa`) from
  `^src/gpu/xmxres\.py$` to `^src/gpu/xmxrt/native\.py$`. No new exemption is
  needed anywhere else; if a moved module trips a rule, the construct is fixed,
  not exempted.

## 9. Invariants

1. **Same pass stream.** For a given configuration and frame size, the ordered
   list of `xmx_rec_*` calls and their arguments is identical before and after.
2. **Same keys.** `graph_key()` and `scratch_key()` are identical for every
   configuration reachable from the environment.
3. **Same bytes.** Frame output is byte-identical (the repo's exactness
   contract: fused passes equal decomposed passes on the same path; nothing in
   this refactor touches either).
4. **Same defaults.** Every switch default, environment name and capability
   gate is unchanged, including the Vulkan-specific `fuse_window_block` once
   that commit lands.
5. **Comments preserved.** Every comment and docstring in `xmxres.py` exists in
   exactly one new module, beside the code it describes.

## 10. Verification

### 10.1 Pass-stream golden (built first, before any move)

A recording library stand-in captures every `xmx_rec_*` / `xmx_*` call (name,
args, with buffer ids normalised to allocation order). A harness records one
frame at 1280x768 through `nr_frame_resident` with that stand-in for:

- the default configuration on each backend name (`vulkan`, `opencl`),
- each switch flipped individually from the default.

Output: a golden file of call streams plus `graph_key`/`scratch_key` per
configuration. Each step of section 11 must reproduce it exactly. This is the
test that catches a dropped flag bit or a reordered argument, which the
numeric tests may not.

### 10.2 Device tests

Per step, on the A770:

- `bash scripts/opencl-test.sh` - 32 pass / 0 fail / 2 skip.
- The Vulkan full suite (`NR_GPU_BACKEND=vulkan`) - 32 / 0 / 2.
- `src/layer/test_daemon.py`, `src/tools/test_release.py`,
  `src/tools/build_check.py`.

### 10.3 Performance

`frame_profile.py --size 1280 768 --runs 6 --seconds 0.5` and the replay bench,
2-3 interleaved runs per backend against the pre-refactor baseline. Python
call overhead per pass changes from a bound-method call to a module-attribute
call; it is expected to be inside noise and is measured, not assumed.

### 10.4 Structure

- An import test asserting the dependency rules in 4.1 (no upward imports, no
  pass module importing another, `ctypes`/`os.environ` only where allowed).
- Repository hooks: file length (no override for any `xmxrt` file), module
  size, banned words.

Every run gets a diary entry in `recomp/diary/dlss-nr.md` with the command,
result and error counts.

## 11. Order of work

Each step is one commit, green on both backends, golden-identical.

0. **Golden.** Add the recording stand-in and the pass-stream harness (10.1);
   capture the golden on the current tree.
1. **Leaf modules.** Create `xmxrt/` with `native`, `ops`, `memory`,
   `scratch`, `profile`; `xmxres.py` imports from them. Update the `.spv`
   scanners, `RUNTIME_MODULES`, `test_release.py`. Hand over the
   banned-words re-scope.
2. **Configuration.** Add `FuseConfig`; `Runtime` holds `self.config`;
   migrate every switch read and write (graph code, tests, benches) to
   `.config`.
3. **Vulkan window-block default** (Run 60) as its own commit on top of 2, in
   `FuseConfig.from_environment`, with its comment. Golden updated for the
   Vulkan default only, and the diff of that golden is the review evidence.
4. **Recorder.** Introduce `Recorder` with allocation/recording; move
   `_Independent`.
5. **Pass families**, one commit each: `gemm`, `attention`, `elementwise`.
   Each moves the functions and migrates every caller of them.
6. **Delete `xmxres.py`.** Migrate the last importers (daemon, `nr_frame.py`,
   tests, benches), update the comments listed in 7.1 and the two docs. Hand
   over the file-length override removal.

Steps 1-5 leave `xmxres.py` as a shrinking module that still works; there is
never a commit where the tree depends on a half-moved symbol.

## 12. Open decisions

1. **How graph code calls passes.**
   - (A, recommended) Module functions: `gemm.gemm(rec, ...)`. The encoders are
     genuinely decoupled from the device object; monkeypatching is per-module;
     ~300 call sites change mechanically.
   - (B) `Recorder` keeps one-line forwarding methods (`rec.gemm(...)`).
     Smaller diff, but `Recorder` keeps ~60 methods of pure indirection and the
     coupling in 2.4.4 stays.
2. **Package vs flat files.** Recommended: `src/gpu/xmxrt/`. Flat `xmx_*.py`
   files in `src/gpu` use 10 of the 12 remaining file slots.
3. **`batch_ffn`, `input_fp16`, `compact_head`, `joint_qkv`, `qkv_epilogue`**
   are not fusions but sit in the same key. Kept in `FuseConfig` (one key, one
   owner); renaming the class to `GraphConfig` is a cosmetic choice left open.

## 13. Risks

| Risk | Mitigation |
|---|---|
| A switch bit lost or shifted in the key - wrong replay with no error anywhere | explicit expression kept verbatim; golden keys for every single-switch flip |
| Argument order changed while moving an encoder | golden pass stream compares arguments, not just call names |
| A bench's monkeypatch no longer intercepts | graph code calls through the module (`gemm.gemm`), never `from gemm import gemm`; covered by the import test |
| Windows package missing a new module | `RUNTIME_MODULES` updated in step 1; `test_release.py` copies the package |
| Comments lost in the move | invariant 9.5; each step's diff is checked for removed comment lines |
| Hidden performance cost of the call shape | 10.3, interleaved against baseline |
| Mid-migration tree that only works on one backend | both suites run on every step |
