# DLSS 5 on Intel Arc

**dlss-on-arc** runs the inference passes of NVIDIA DLSS 5 on Intel Arc GPUs through Vulkan
`VK_KHR_cooperative_matrix`, with a Vulkan layer that applies them to a game at present time.

The goal is DLSS 5 on both Arc generations, Alchemist and Battlemage, covering Neural
Rendering, upscaling and frame generation. Start with [Getting Started](#getting-started).

## Scope

| Pass | Status |
| --- | --- |
| Neural Rendering (DLSS-NR) | Working: live in games on Linux and Windows |
| Upscaling (Super Resolution) | Planned |
| Frame generation | Work in progress |

| GPU | Matrix shape | Status |
| --- | --- | --- |
| Battlemage and Lunar Lake (Xe2) | 8x16x16 | Supported; measured on Arc 140V, B580 reports the same configurations |
| Alchemist (Xe-HPG, A-series) | 8x8x16 | Target; the current kernels need 8x16x16 |
| Any Vulkan 1.3 GPU without cooperative matrix | none | Portable multiply-add GEMM path |

Metal (macOS) and Direct3D 12 runtimes implement the same entry points. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the recovered network.

## Getting Started

### Prerequisites

- An Intel Arc GPU from the table above. `src/probe/coopmat_probe.c` reports what your GPU
  offers.
- Your own copy of `nvngx_dlssnr.dll`. The repository contains no NVIDIA binaries and no
  weights.
- Linux: a C compiler, `glslangValidator`, the Vulkan loader, libpng, Python 3 with NumPy.
- Windows: the release needs a 64-bit Python and your DLL; see
  [docs/RELEASE-QUICKSTART.md](docs/RELEASE-QUICKSTART.md). Building from source:
  [docs/WINDOWS-QUICKSTART.md](docs/WINDOWS-QUICKSTART.md).

### Build (Linux)

```sh
mkdir -p work
git clone --depth 1 --branch v1.4.321 \
          https://github.com/KhronosGroup/Vulkan-Headers.git work/vulkan-headers
git clone https://github.com/iamwavecut/MLX-DLSS.git work/mlx-dlss
git -C work/mlx-dlss checkout 06a3e11a8b68817127406ace5c764463543f699b
make
```

CMake builds the same targets on Linux, macOS, Windows and Android:
`cmake -S . -B build && cmake --build build && ctest --test-dir build`.

### Extract the weights

```sh
mkdir -p work/mlxw
uv run python work/mlx-dlss/python/mlxdlss/tools/extract_dlssnr_weights.py \
        /path/to/nvngx_dlssnr.dll work/mlxw/dlssnr-packed.safetensors
uv run python work/mlx-dlss/python/mlxdlss/tools/unpack_dlssnr_weights.py \
        work/mlxw/dlssnr-packed.safetensors work/mlxw/dlssnr-logical.safetensors
```

### Run it in a game

```sh
uv run python src/layer/nr_daemon.py --settings /tmp/nr_settings.json
NR_LIVE=1 src/layer/nr-photo --steam <appid>   # prints the Steam launch option
```

Generate the launch option in your own clone: it carries the absolute path to your
`work/layer`. Start with a 640x360 to 800x450 game window and a render scale near 0.4.

## How it works

```mermaid
flowchart LR
  game([Game presents a frame]) --> layer[Vulkan layer captures the frame]
  layer --> daemon[Daemon runs the network on the GPU]
  daemon --> compose[Compose the result over the game frame]
  compose --> screen(Frame on screen)
```

The layer hooks `vkQueuePresentKHR` and hands each frame to a daemon over a Unix socket on
Linux or a named pipe on Windows. The daemon keeps the weights resident and runs every GEMM
in FP16 with FP32 accumulate. DirectX 8 to 11 games reach the layer through DXVK; on Linux,
Proton also covers DirectX 12 through VKD3D-Proton.

Runtime settings and measured frame times: [docs/SETTINGS.md](docs/SETTINGS.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/ref/` | CPU reference: graph, features, composition, temporal path |
| `src/gpu/` | Compute runtimes: Vulkan, Metal, Direct3D 12 |
| `src/layer/` | Vulkan layer, daemon and control tools |
| `src/bench/` | Measurement programs |
| `src/probe/` | GPU capability probes |
| `src/tools/` | DLL and weight readers, publishing checks |
| `docs/` | Architecture specification and setup guides |
| `dist-tools/`, `scripts/`, `tools/` | Windows setup and release tooling |

## Contributing

```sh
make check        # lint, type-check, test
make check-push   # pre-push gate
sudo make install-hooks
```

Hooks are mandatory (pre-commit, commit-msg, pre-push); the check list is in
`/opt/workspace-ci/README.md`. Tests that need weights skip without them. The
lint and type-check gates run ruff and mypy; the rules still switched off, and
how to turn them back on, are in [docs/QUALITY-GATES.md](docs/QUALITY-GATES.md).

## Provenance and licences

Apache License 2.0: [LICENSE](LICENSE) and [NOTICE](NOTICE).

- Forked from [uzbekunknown/dlss-nr-on-intel](https://github.com/Uzbekunknown/dlss-nr-on-intel)
  (Apache-2.0) through [andyvand/dlss-nr-on-vulkan](https://github.com/andyvand/dlss-nr-on-vulkan).
  Their histories hold the original authorship.
- The network graph was recovered by [MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS)
  (Apache-2.0) and corrected against [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR)
  (MIT).
- The Windows release carries [DXVK](https://github.com/doitsujin/dxvk) under its zlib licence.
  Vulkan-Headers (Apache-2.0) and bin2c (BSD-2) are fetched at build time.

DLSS, Neural Rendering and `nvngx_dlssnr.dll` belong to NVIDIA Corporation. This project is
an independent reimplementation, unaffiliated with NVIDIA, and distributes no NVIDIA code,
binaries or weights.
