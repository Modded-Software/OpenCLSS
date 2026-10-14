"""Release layout constants, the launch profile, and the shared JSON helpers.

Importing this module needs only Python's standard library.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
OWNER_NAME = ".windows-wizard-owner.json"
LIBRARIES = ("libxmx.dll", "libnr_image.dll", "libnr_alloc.dll")
SHADERS = frozenset(
    (
        "attention.spv",
        "attention_rows.spv",
        "ffn_fused.spv",
        "gemm_batched.spv",
        "gemm_coopmat.spv",
        "gemm_resident.spv",
        "gemm_staged.spv",
        "gemm_staged32.spv",
        "gemm_staged32_deep.spv",
        "gemm_staged_int8.spv",
        "gemm_tiled.spv",
        "global_attention.spv",
        "half_probe.spv",
        "history.spv",
        "resident.spv",
        "window_attention.spv",
        "window_block.spv",
    )
)
SOURCE_DIRS = ("layer", "ref", "gpu", "bench")
# The layer the game's own architecture loads; the daemon and its libraries are always x64.
LAYERS = {"x64": "nr_layer.dll", "x86": "nr_layer32.dll"}
# DXVK as its release lays it out, x64/ and x32/, and the files it puts beside a game.
DXVK_DIRS = {"x64": "x64", "x86": "x32"}
DXVK_FILES = ("d3d8.dll", "d3d9.dll", "d3d10core.dll", "d3d11.dll", "dxgi.dll")
# Beside every game as vulkan-1.dll: it gives the game NR's environment however the game is
# started (nr_vulkan_proxy.c), from ENV_FILE in the installation.
PROXIES = {"x64": "nr_vulkan_proxy.dll", "x86": "nr_vulkan_proxy32.dll"}
PROXY_NAME = "vulkan-1.dll"
ENV_FILE = "nr-env.txt"
# Where the proxy lists the folders the game was started in, for DXVK's first log there.
START_FOLDERS = "start-folders.txt"
# In the game's dlss-nr folder: what NR put beside the game, and the game's own files it set
# aside to do so, which Remove NR puts back.
GAME_FILES = "game-files.json"
GAME_BACKUP = "game-backup"
RUNTIME_MODULES = (
    "src/layer/nr_daemon.py",
    "src/layer/nr_alloc.py",
    "src/layer/nr_pipe.py",
    "src/layer/nr_paths.py",
    "src/layer/nr_knobs.py",
    "src/ref/nr_frame.py",
    "src/ref/nr_model.py",
    "src/ref/nr_image.py",
    "src/ref/image_io.py",
    "src/gpu/xmx.py",
    "src/gpu/xmxrt/__init__.py",
    "src/gpu/xmxrt/ops.py",
    "src/gpu/xmxrt/native.py",
    "src/gpu/xmxrt/memory.py",
    "src/gpu/xmxrt/scratch.py",
    "src/gpu/xmxrt/profile.py",
    "src/gpu/xmxrt/config.py",
    "src/gpu/xmxrt/recorder.py",
    "src/gpu/xmxrt/gemm.py",
    "src/gpu/xmxrt/attention.py",
    "src/gpu/xmxrt/elementwise.py",
    "scripts/get_weights.py",
    "work/mlx-dlss/python/mlxdlss/features.py",
    "work/mlx-dlss/python/mlxdlss/tools/extract_dlssnr_weights.py",
    "work/mlx-dlss/python/mlxdlss/tools/unpack_dlssnr_weights.py",
    "LICENSE",
    "NOTICE",
    "work/mlx-dlss/LICENSE",
)
MAX_OUTPUT = 256 * 1024
MAX_LOG_READ = 1024 * 1024
MAX_LOG_TAIL = 64 * 1024
VC_REDIST = "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist"
# The DLL every test here extracted its weights from (notes/CLAUDE.md, Phase 0).
TESTED_DLL_VERSION = "310.8.0.0"
TESTED_DLL_SHA256 = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
WEIGHTS_LOG = "work/logs/get-weights.log"


@dataclass
class Profile:
    root: Path
    python: Path
    game_exe: Path
    dll: Path | None = None
    api: str = "vulkan"
    game_args: list[str] = field(default_factory=list)
    disable_fossilize: bool = True
    launch_mode: str = "direct"
    steam_app_id: str = ""

    def __post_init__(self):
        self.root = Path(self.root).expanduser().resolve()
        self.python = Path(self.python).expanduser().resolve()
        self.game_exe = Path(self.game_exe).expanduser().resolve()
        self.dll = Path(self.dll).expanduser().resolve() if self.dll else None
        self.game_args = list(self.game_args)

    def to_dict(self):
        return {
            "root": str(self.root),
            "python": str(self.python),
            "game_exe": str(self.game_exe),
            "dll": str(self.dll) if self.dll else None,
            "api": self.api,
            "game_args": list(self.game_args),
            "disable_fossilize": self.disable_fossilize,
            "launch_mode": self.launch_mode,
            "steam_app_id": self.steam_app_id,
        }

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Profile must be a JSON object")
        arguments = values.get("game_args", [])
        if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
            raise ValueError("game_args must be a list of strings")
        fossilize = values.get("disable_fossilize", True)
        if not isinstance(fossilize, bool):
            raise ValueError("disable_fossilize must be a boolean")
        return cls(
            root=Path(values.get("root") or Path(*Path(__file__).resolve().parts[:-4])),
            python=Path(values.get("python") or sys.executable),
            game_exe=Path(values.get("game_exe") or "."),
            dll=Path(values["dll"]) if values.get("dll") else None,
            api=values.get("api", "vulkan"),
            game_args=arguments,
            disable_fossilize=fossilize,
            launch_mode=values.get("launch_mode", "direct"),
            steam_app_id=str(values.get("steam_app_id") or ""),
        )


def _read_json(path, limit=1024 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"JSON file is too large: {path}")
    return json.loads(raw.decode("utf-8-sig"))


def _atomic_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".nr-json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(values, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _canonical(path):
    return os.path.normcase(str(Path(path).resolve())).replace("\\", "/").rstrip("/")
