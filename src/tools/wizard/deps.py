"""Local model inputs, release files, preflight checks and game-folder staging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from wizard import config, process, state


def _weights_state(profile):
    path = profile.root / "work/mlxw/dlssnr-logical.safetensors"
    try:
        with path.open("rb") as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise ValueError("missing safetensors header")
            length = int.from_bytes(size_bytes, "little")
            if not 2 <= length <= 16 * 1024 * 1024:
                raise ValueError("invalid safetensors header length")
            header = json.loads(handle.read(length))
        data_size = path.stat().st_size - 8 - length
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if len(tensors) != 649 or data_size < 0:
            raise ValueError("expected 649 logical tensors")
        for value in tensors.values():
            offsets = value.get("data_offsets", [])
            if (
                len(offsets) != 2
                or not all(isinstance(v, int) for v in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_size
            ):
                raise ValueError("invalid tensor data offsets")
        return {"ready": True, "path": str(path), "detail": "649 local logical tensors"}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {"ready": False, "path": str(path), "detail": str(error)}


def _owned_installation(profile):
    path = state.installed_path(profile)
    if not path.exists():
        return True, "New game-local installation"
    if path.is_symlink() or (os.name == "nt" and path.is_junction()):
        return False, "Installation directory cannot be a link or junction"
    if not path.is_dir():
        return False, "A file blocks the installation directory"
    try:
        owner = config._read_json(path / config.OWNER_NAME, limit=16 * 1024)
        matches = owner.get("schema_version") == config.SCHEMA_VERSION and (
            config._canonical(owner.get("root", "")) == config._canonical(profile.root)
        )
    except (OSError, ValueError, AttributeError):
        matches = False
    return (
        matches,
        "Owned by this release"
        if matches
        else "Existing dlss-nr directory is not owned by this release",
    )


def _file_version(path):
    """The file version Windows' Properties shows, from the version resource, or None.

    GetFileVersionInfoW maps the file as data; no code in it runs."""
    if os.name != "nt" or not path or not Path(path).is_file():
        return None
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version")
    version.GetFileVersionInfoSizeW.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    version.GetFileVersionInfoW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    )
    version.VerQueryValueW.argtypes = (
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.UINT),
    )
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    pointer, length = ctypes.c_void_p(), wintypes.UINT()
    if (
        not version.GetFileVersionInfoW(str(path), 0, size, data)
        or not version.VerQueryValueW(data, "\\", ctypes.byref(pointer), ctypes.byref(length))
        or length.value < 52
    ):
        return None
    # VS_FIXEDFILEINFO: its signature, the structure's version, then the file's as two DWORDs.
    fixed = ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD * 4)).contents
    if fixed[0] != 0xFEEF04BD:
        return None
    return f"{fixed[2] >> 16}.{fixed[2] & 0xFFFF}.{fixed[3] >> 16}.{fixed[3] & 0xFFFF}"


def _dll_info(path):
    """The user's DLL as a report gives it: its version and whether it is the one tested."""
    if not path or not Path(path).is_file():
        return {"present": False}
    digest = _sha256(path)
    return {
        "present": True,
        "size": Path(path).stat().st_size,
        "version": _file_version(path),
        "sha256": digest,
        "tested": digest == config.TESTED_DLL_SHA256,
    }


def _load_dependencies(profile, timeout=30):
    # libnr_image.dll's one dependency outside Windows is VCOMP140.DLL, MSVC's OpenMP, which
    # the Visual C++ Redistributable installs: asked for by name, so its absence says so.
    code = """
import ctypes, json, os, sys
paths = json.loads(sys.argv[1]); result = {'loaded': [], 'failures': {}}
handles = [os.add_dll_directory(os.path.dirname(path)) for path in paths]
for path in ['vulkan-1.dll'] + paths:
    try:
        ctypes.WinDLL(path)
        result['loaded'].append(path)
    except OSError as error:
        result['failures'][path] = str(error)
if result['failures']:
    try:
        ctypes.WinDLL('vcomp140.dll')
    except OSError:
        result['failures']['VCOMP140.DLL'] = 'missing'
print(json.dumps(result))
"""
    got = process._run(
        [
            profile.python,
            "-I",
            "-c",
            code,
            json.dumps([str(profile.root / "work" / name) for name in config.LIBRARIES]),
        ],
        timeout=timeout,
    )
    try:
        result = json.loads(got["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {
            "failures": {
                "dependency_probe": f"no answer in {timeout} s"
                if got["timed_out"]
                else got["output"] or "No probe output"
            }
        }
    if "VCOMP140.DLL" in result.get("failures", {}):
        result["failures"]["VCOMP140.DLL"] = (
            "missing: install the Microsoft Visual C++ x64 Redistributable, " + config.VC_REDIST
        )
    result["ok"] = got["returncode"] == 0 and not got["timed_out"] and not result.get("failures")
    return result


def validate(profile):
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    check(
        "release_root",
        profile.root.is_dir() and str(profile.root).isascii(),
        "The permanent release root must exist and use an ASCII path",
    )
    check(
        "api",
        profile.api in ("vulkan", "dxvk"),
        "Native Vulkan or an existing x64 DXVK configuration",
    )
    check(
        "launch_mode",
        profile.launch_mode in ("direct", "steam")
        and (
            profile.launch_mode != "steam"
            or not profile.steam_app_id
            or profile.steam_app_id.isdecimal()
        ),
        "Direct launch, or Steam with an optional numeric app ID; blank uses autodetection",
    )
    check("python_file", profile.python.is_file(), profile.python)
    info = process._python_info(profile.python) if profile.python.is_file() else {}
    check(
        "python_native_x64",
        process._native_python(info),
        "This Python did not answer: " + info["probe_error"]
        if info.get("probe_error")
        else f"{info.get('platform', 'unknown')} / {info.get('bits', '?')} bit; {info.get('version', '')}",
    )
    check(
        "python_packages",
        not info.get("package_errors")
        and all(name in info.get("packages", {}) for name in ("numpy", "safetensors")),
        info.get("package_errors") or info.get("packages") or "NumPy and safetensors are required",
    )
    arch = process.pe_architecture(profile.game_exe) if profile.game_exe.is_file() else None
    check(
        "game_architecture",
        arch in config.LAYERS,
        f"{profile.game_exe}: {arch or 'not a PE executable'}; 64-bit and 32-bit games are supported",
    )
    layer = config.LAYERS.get(arch, config.LAYERS["x64"])
    proxy = config.PROXIES.get(arch, config.PROXIES["x64"])
    shader_names = process._shader_names(profile.root)
    required = [
        layer,
        proxy,
        *config.RUNTIME_MODULES,
        *["work/" + name for name in (*config.LIBRARIES, *shader_names)],
    ]
    missing = [name for name in required if not (profile.root / name).is_file()]
    missing.extend(
        "src/" + name for name in config.SOURCE_DIRS if not (profile.root / "src" / name).is_dir()
    )
    check(
        "runtime_files",
        not missing,
        "Missing: " + ", ".join(missing)
        if missing
        else "Complete local release runtime and extractors",
    )
    bad_arch = [
        "work/" + value
        for value in config.LIBRARIES
        if process.pe_architecture(profile.root / "work" / value) != "x64"
    ]
    for name in (layer, proxy):
        if (
            arch in config.LAYERS
            and (profile.root / name).is_file()
            and process.pe_architecture(profile.root / name) != arch
        ):
            bad_arch.append(name)
    check(
        "runtime_architecture",
        not bad_arch,
        "Wrong architecture: " + ", ".join(bad_arch)
        if bad_arch
        else f"Native libraries are x64; {layer} and {proxy} for this {arch or 'unidentified'} game",
    )
    dependency = (
        _load_dependencies(profile)
        if process._native_python(info) and not bad_arch
        else {
            "ok": False,
            "failures": {"probe": "Requires native x64 Python and x64 runtime"},
        }
    )
    failures = dependency.get("failures") or {}
    check(
        "runtime_dependencies",
        dependency.get("ok"),
        "; ".join(f"{Path(name).name}: {why}" for name, why in failures.items())
        or "Vulkan loader and native DLL dependencies load; no GPU work performed",
    )
    weights = _weights_state(profile)
    dll_ok = bool(
        profile.dll
        and profile.dll.is_file()
        and profile.dll.name.lower() == "nvngx_dlssnr.dll"
        and process.pe_architecture(profile.dll) == "x64"
    )
    if weights["ready"]:
        model_detail = weights["detail"]
    elif dll_ok:
        version = _file_version(profile.dll)
        model_detail = (
            f"Install extracts the weights from {profile.dll}, version {version or 'unknown'}"
            + (
                ""
                if version == config.TESTED_DLL_VERSION
                else f"; tested with {config.TESTED_DLL_VERSION}"
            )
        )
    else:
        model_detail = "Supply your own x64 nvngx_dlssnr.dll or 649-tensor local logical weights"
    check("local_model_input", weights["ready"] or dll_ok, model_detail)
    owns, detail = _owned_installation(profile)
    check("installation_target", owns, detail)
    for name in ("windows-profile.json", "nr_settings.json", "nr_trigger"):
        path = profile.root / "work" / name
        check(
            "state_path_" + name,
            not path.exists() or (path.is_file() and not path.is_symlink()),
            path,
        )
    settings = profile.root / "work/nr_settings.json"
    try:
        settings_ok = not settings.exists() or isinstance(config._read_json(settings), dict)
    except (OSError, ValueError):
        settings_ok = False
    check("settings_format", settings_ok, "Existing settings must be a JSON object")
    template = profile.root / "VkLayer_dlss_nr.json"
    if not template.is_file():
        template = profile.root / "src/layer/VkLayer_dlss_nr.json"
    try:
        manifest = config._read_json(template)
        manifest_ok = (
            isinstance(manifest.get("layer"), dict)
            and manifest["layer"].get("name") == "VK_LAYER_dlssnr_intel"
        )
    except (OSError, ValueError, AttributeError):
        manifest_ok = False
    check(
        "manifest",
        manifest_ok,
        "A valid layer manifest must be present before installation",
    )
    if profile.api == "dxvk":
        folder = profile.root / "dxvk" / config.DXVK_DIRS.get(arch, "x64")
        wrong = [
            name for name in config.DXVK_FILES if process.pe_architecture(folder / name) != arch
        ]
        check(
            "dxvk_files",
            arch in config.LAYERS and not wrong,
            f"DXVK for this {arch} game, from {folder}"
            if arch in config.LAYERS and not wrong
            else "The release's DXVK is missing or of another architecture: " + ", ".join(wrong),
        )
    files_ok, files_detail = _game_files_check(profile, arch)
    check("game_files", files_ok, files_detail)
    ok = all(item["ok"] for item in checks)
    return {
        "ok": ok,
        "error": None if ok else "Preflight checks failed",
        "checks": checks,
        "python_info": info,
        "weights": weights,
        "runtime_dependencies": dependency,
    }


def ensure_dependencies(profile, emit=None):
    info = process._python_info(profile.python)
    if not process._native_python(info):
        return {
            "ok": False,
            "error": "Choose native x64 Windows Python before preparing dependencies",
            "output": info.get("output", ""),
        }
    directory = profile.root / "work/windows-python"
    interpreter = directory / "Scripts/python.exe"
    commands = []
    try:
        if directory.exists() and (
            directory.is_symlink() or not (directory / "pyvenv.cfg").is_file()
        ):
            raise ValueError(
                "Existing windows-python directory is not a local Python virtual environment"
            )
        if not interpreter.is_file():
            directory.parent.mkdir(parents=True, exist_ok=True)
            if emit:
                emit("Creating release-local Python environment\n")
            got = process._run(
                [profile.python, "-I", "-m", "venv", str(directory)],
                timeout=120,
                emit=emit,
            )
            commands.append(got)
            if got["returncode"] != 0 or got["timed_out"]:
                raise RuntimeError("Could not create the local virtual environment")
        if emit:
            emit("Installing NumPy and safetensors into the local environment\n")
        got = process._run(
            [
                interpreter,
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "numpy",
                "safetensors",
            ],
            timeout=600,
            emit=emit,
        )
        commands.append(got)
        if got["returncode"] != 0 or got["timed_out"]:
            raise RuntimeError(
                "Dependency installation failed; the global interpreter was not modified"
            )
        chosen = replace(profile, python=interpreter)
        chosen_info = process._python_info(chosen.python)
        if (
            not process._native_python(chosen_info)
            or chosen_info.get("package_errors")
            or not all(name in chosen_info.get("packages", {}) for name in ("numpy", "safetensors"))
        ):
            raise RuntimeError("The local interpreter did not pass its dependency checks")
        return {
            "ok": True,
            "error": None,
            "python": str(chosen.python),
            "profile": chosen.to_dict(),
            "output": "\n".join(value["output"] for value in commands),
            "commands": commands,
        }
    except (OSError, ValueError, RuntimeError) as error:
        return {
            "ok": False,
            "error": str(error),
            "output": "\n".join(value["output"] for value in commands),
            "commands": commands,
        }


def _copy_runtime(profile, stage, destination):
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.safetensors", "nvngx_dlssnr.dll", ".git"
    )
    for name in config.SOURCE_DIRS:
        shutil.copytree(
            profile.root / "src" / name,
            stage / "src" / name,
            dirs_exist_ok=True,
            ignore=ignore,
        )
    shutil.copytree(
        profile.root / "work/mlx-dlss",
        stage / "work/mlx-dlss",
        dirs_exist_ok=True,
        ignore=ignore,
    )
    arch = process.pe_architecture(profile.game_exe)
    layer = config.LAYERS[arch]
    shutil.copy2(profile.root / layer, stage / layer)
    for name in (*config.LIBRARIES, *process._shader_names(profile.root)):
        shutil.copy2(profile.root / "work" / name, stage / "work" / name)
    template = profile.root / "VkLayer_dlss_nr.json"
    if not template.is_file():
        template = profile.root / "src/layer/VkLayer_dlss_nr.json"
    manifest = config._read_json(template)
    manifest["layer"]["library_path"] = str(destination / layer)
    manifest["layer"]["library_arch"] = "64" if arch == "x64" else "32"
    config._atomic_json(stage / "VkLayer_dlss_nr.json", manifest)
    config._atomic_json(
        stage / config.OWNER_NAME,
        {
            "schema_version": config.SCHEMA_VERSION,
            "root": str(profile.root),
            "game_exe": str(profile.game_exe),
        },
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dxvk_logs(profile):
    """The logs DXVK writes into the folder a game was started in, without DXVK_LOG_PATH."""
    return [f"{profile.game_exe.stem}_{Path(module).stem}.log" for module in config.DXVK_FILES]


def _game_record(folder):
    """What NR put beside the game, from an installation folder (or its stage)."""
    path = Path(folder) / config.GAME_FILES
    if not path.exists():
        return {"schema_version": config.SCHEMA_VERSION, "arch": None, "files": {}}
    record = config._read_json(path, limit=64 * 1024)
    if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
        raise ValueError(f"{path} is not a record of placed game files")
    return record


def _game_sources(profile, arch):
    """What goes beside the game, by name: the proxy for every game, and DXVK for DirectX 8-11."""
    sources = {config.PROXY_NAME: profile.root / config.PROXIES[arch]}
    if profile.api == "dxvk":
        folder = profile.root / "dxvk" / config.DXVK_DIRS[arch]
        sources.update((name, folder / name) for name in config.DXVK_FILES)
    return sources


def _game_files_check(profile, arch):
    """Whether installing may touch the game's folder, and which of its files it would set aside."""
    try:
        record = _game_record(state.installed_path(profile))
    except (OSError, ValueError) as error:
        return False, str(error)
    if arch not in config.LAYERS:
        return True, "Nothing is put beside the game"
    sources = _game_sources(profile, arch)
    placed = [name for name, entry in record["files"].items() if entry.get("placed")]
    # An installation from before the proxy has fewer files placed, and gains it; one with
    # DXVK's files placed for DirectX cannot become a Vulkan one without Remove NR first.
    if placed and (not set(placed) <= set(sources) or record.get("arch") != arch):
        return (
            False,
            "Remove NR from this game before changing its graphics API or executable",
        )
    aside = []
    for name, source in sources.items():
        target = profile.game_exe.parent / name
        entry = record["files"].get(name, {})
        if target.is_symlink() or (target.exists() and not target.is_file()):
            return False, f"{target} is not a regular file"
        if (
            target.is_file()
            and not entry.get("placed")
            and source.is_file()
            and _sha256(target) != _sha256(source)
        ):
            aside.append(name)
    return True, (
        "Set aside, and put back by Remove NR: " + ", ".join(aside)
        if aside
        else "No file of the game's is replaced"
    )


def _place_game_files(profile, arch, folder):
    """Put the proxy, and DXVK for DirectX 8-11, beside the game, each step recorded in
    `folder`'s GAME_FILES first.

    A file of the same name that NR did not put there is moved into `folder`'s GAME_BACKUP
    and comes back with Remove NR; an identical copy someone else put there is left alone.
    `folder` is the installation's stage, in the game's own folder, so every move is a
    rename. Returns what was done, for _undo_game_files; undoes its own part if it fails."""
    game = profile.game_exe.parent
    folder = Path(folder)
    record = _game_record(folder)
    record["arch"] = arch
    if "logs_before" not in record:
        # DXVK's logs already beside the game stay when NR is removed; only later ones go.
        record["logs_before"] = [name for name in _dxvk_logs(profile) if (game / name).exists()]
    # And in the folders the game is started in, those written before this.
    record.setdefault("installed_time", time.time())
    files = record["files"]
    done = []
    try:
        for name, source in _game_sources(profile, arch).items():
            target = game / name
            ours = _sha256(source)
            entry = files.get(name, {})
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"{target} is not a regular file")
            if target.is_file():
                current = _sha256(target)
                if current == ours:
                    if not entry.get("placed"):
                        files[name] = {"placed": False, "sha256": ours}
                        config._atomic_json(folder / config.GAME_FILES, record)
                    continue
                if entry.get("placed") and current == entry.get("sha256"):
                    # NR's own copy from an earlier release: kept aside until this one commits.
                    undo = folder / ".undo"
                    undo.mkdir(exist_ok=True)
                    os.replace(target, undo / name)
                    done.append((name, "replaced", dict(entry)))
                else:
                    backup = folder / config.GAME_BACKUP
                    backup.mkdir(exist_ok=True)
                    if (backup / name).exists():
                        raise ValueError(f"{backup / name} already holds a set-aside file")
                    os.replace(target, backup / name)
                    entry = {"backup": True, "previous_sha256": current}
                    done.append((name, "set-aside", None))
            else:
                done.append((name, "added", None))
            temporary = game / (name + ".dlss-nr-new")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            files[name] = {
                "placed": True,
                "sha256": ours,
                "backup": bool(entry.get("backup")),
                "previous_sha256": entry.get("previous_sha256"),
            }
            config._atomic_json(folder / config.GAME_FILES, record)
    except BaseException:
        _undo_game_files(profile, folder, done)
        raise
    return done


def _write_proxy_environment(profile, stage, destination):
    """ENV_FILE, which the proxy reads in the game's process: the layer's environment, the game
    it is for, where to record a launch, and the game's own loader if one was set aside."""
    lines = [
        "# Read by vulkan-1.dll beside the game (nr_vulkan_proxy.c) in the game's own process.",
        "NR_GAME_EXE=" + str(profile.game_exe),
        "NR_LAUNCH_STATE=" + str(profile.root / "work/windows-wizard/launch-state.json"),
    ]
    if _game_record(stage)["files"].get(config.PROXY_NAME, {}).get("backup"):
        lines.append("NR_REAL_VULKAN=" + str(destination / config.GAME_BACKUP / config.PROXY_NAME))
    lines += [f"{name}={value}" for name, value in state.layer_environment(profile).items()]
    (stage / config.ENV_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _undo_game_files(profile, folder, done):
    """Reverse _place_game_files' steps, last first, with the record where `folder` now is."""
    game = profile.game_exe.parent
    folder = Path(folder)
    for name, kind, entry in reversed(done):
        target = game / name
        target.unlink(missing_ok=True)
        if kind == "set-aside" and (folder / config.GAME_BACKUP / name).is_file():
            os.replace(folder / config.GAME_BACKUP / name, target)
        elif kind == "replaced" and (folder / ".undo" / name).is_file():
            os.replace(folder / ".undo" / name, target)
    try:
        record = _game_record(folder)
        for name, kind, entry in done:
            if kind == "replaced":
                record["files"][name] = entry
            else:
                record["files"].pop(name, None)
        config._atomic_json(folder / config.GAME_FILES, record)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"[wizard] cannot record the game files: {error}\n")
