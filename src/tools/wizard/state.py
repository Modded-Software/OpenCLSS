"""The saved launch profile, the runtime settings document, and the launch environment."""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

from wizard import config


def profile_path(root):
    return Path(root) / "work/windows-profile.json"


def _runtime_knobs():
    """Read the same standard-library-only catalogue used by Linux nr-panel.

    The backend lives in src/tools in a checkout and in scripts in a release.
    Neither location needs NumPy, a selected game or a running daemon to describe
    the controls. Keep the panel's normal slider ranges distinct from the wider
    ranges accepted by the daemon, so existing advanced values survive a refresh.
    """
    here = Path(*Path(__file__).resolve().parts[:-3])
    candidates = (here / "layer/nr_knobs.py", here / "src/layer/nr_knobs.py")
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise ValueError("The release is missing its runtime control catalogue")
    sys.path.insert(0, str(source.parent))
    try:
        import nr_knobs as module
    finally:
        sys.path.remove(str(source.parent))
    knobs = []
    for knob in module.KNOBS:
        high = {"min_extent": 4096.0, "release": 255.0}.get(knob.name, knob.high)
        knobs.append(
            {
                "name": knob.name,
                "label": knob.label,
                "kind": knob.kind,
                "low": knob.low,
                "high": knob.high,
                "step": knob.step,
                "default": knob.default,
                "summary": knob.summary,
                "detail": knob.detail,
                "runtime_low": knob.low,
                "runtime_high": high,
                "choices": list(module.PROFILES) if knob.kind == "choice" else None,
            }
        )
    return knobs


def settings_path(root):
    """Only the settings belonging to this release, never an inherited NR_ROOT."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The release root does not exist")
    work = root / "work"
    path = work / "nr_settings.json"
    if (
        work.is_symlink()
        or (os.name == "nt" and work.is_junction())
        or (work.exists() and not work.is_dir())
        or config._canonical(work) != config._canonical(root) + "/work"
    ):
        raise ValueError("The release work directory is not an owned regular directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("The release settings path is not an owned regular file")
    return path


def _settings_document(root):
    path = settings_path(root)
    values = config._read_json(path) if path.exists() else {}
    if not isinstance(values, dict):
        raise ValueError("Settings must be a JSON object")
    return path, values


def _validated_settings(values, knobs):
    effective = {}
    for knob in knobs:
        name = knob["name"]
        value = values.get(name, knob["default"])
        if knob["kind"] == "choice":
            if value not in knob["choices"]:
                raise ValueError(f"{name} must be one of {', '.join(knob['choices'])}")
        else:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number, not a boolean")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not math.isfinite(value) or not knob["runtime_low"] <= value <= knob["runtime_high"]:
                raise ValueError(
                    f"{name} must be between {knob['runtime_low']:g} and {knob['runtime_high']:g}"
                )
        effective[name] = value
    return effective


def _settings_result(path, values, knobs):
    effective = _validated_settings(values, knobs)
    return {
        "ok": True,
        "error": None,
        "path": str(path),
        "settings": effective,
        "knobs": knobs,
        "defaults": {knob["name"]: knob["default"] for knob in knobs},
        "explicit_keys": [knob["name"] for knob in knobs if knob["name"] in values],
        "warnings": [],
        "apply": "next_frame",
        "game_restart_required": False,
        "processing_confirmed": False,
    }


def get_settings(root):
    """Describe validated saved values; this is not a daemon acknowledgement."""
    path, values = _settings_document(root)
    return _settings_result(path, values, _runtime_knobs())


def save_settings(root, changes):
    """Merge only changed controls with the latest document and replace atomically.

    Unknown existing keys belong to other tools and are retained. A settings save
    does not install a game, save a launch profile, start NR or change its trigger.
    """
    if not isinstance(changes, dict):
        raise ValueError("Settings changes must be a JSON object")
    knobs = _runtime_knobs()
    known = {knob["name"] for knob in knobs}
    unknown = set(changes) - known
    if unknown:
        raise ValueError("Unknown runtime controls: " + ", ".join(sorted(unknown)))
    path, values = _settings_document(root)
    values.update(changes)
    effective = _validated_settings(values, knobs)
    # Store numbers as numbers even when an existing command-line tool supplied
    # a numeric string; never write bool, NaN or Infinity for a runtime control.
    for name in changes:
        values[name] = effective[name]
    if changes:
        config._atomic_json(path, values)
    return _settings_result(path, values, knobs)


def reset_settings(root, names=None):
    """Write explicit daemon defaults so a live daemon really changes its values.

    Deleting a key cannot reset a running daemon: Settings.refresh deliberately
    keeps its previous value when that key is absent from the next document.
    """
    knobs = _runtime_knobs()
    defaults = {knob["name"]: knob["default"] for knob in knobs}
    if names is None:
        names = list(defaults)
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("Reset controls must be a list of names")
    unknown = set(names) - defaults.keys()
    if unknown:
        raise ValueError("Unknown runtime controls: " + ", ".join(sorted(unknown)))
    return save_settings(root, {name: defaults[name] for name in names})


def load_profile(root):
    path = profile_path(root)
    if not path.exists():
        return None
    profile = config.Profile.from_dict(config._read_json(path))
    if config._canonical(profile.root) != config._canonical(root):
        raise ValueError("Saved profile belongs to a different release root")
    return profile


def save_profile(profile):
    # A Steam configuration an earlier setup left does not hold the profile: its wrapper
    # starts the game from a snapshot of its own (windows_launch._ensure_steam_profile).
    config._atomic_json(profile_path(profile.root), profile.to_dict())
    return {
        "ok": True,
        "error": None,
        "path": str(profile_path(profile.root)),
        "profile": profile.to_dict(),
    }


def installed_path(profile):
    return profile.game_exe.parent / "dlss-nr"


def _clean_environment():
    values = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("NR_", "XMX_", "VK_"))
        and key.upper() not in ("ENABLE_NR_LAYER", "DISABLE_NR_LAYER")
    }
    # A release's own choice is explicit; other Steam layers retain
    # their normal implicit-loader configuration, including the overlay.
    values.pop("DISABLE_VK_LAYER_VALVE_steam_fossilize_1", None)
    return values


def layer_environment(profile):
    """NR's own variables: what a launch adds to the environment, and what the proxy sets."""
    token = hashlib.sha256(config._canonical(profile.root).encode("utf-8")).hexdigest()[:8]
    work = profile.root / "work"
    values = {
        "NR_ROOT": str(profile.root),
        "NR_PYTHON": str(profile.python),
        "NR_DAEMON": str(profile.root / "src/layer/nr_daemon.py"),
        "NR_SETTINGS": str(work / "nr_settings.json"),
        "NR_LAYER_LOG": str(work / "nr_daemon.log"),
        "NR_LAYER_TRIGGER": str(work / "nr_trigger"),
        "NR_LAYER_SOCKET": rf"\\.\pipe\nr_windows_{token}",
        "NR_LAYER_LIVE": "1",
        "NR_LAYER_SPAWN": "1",
        "NR_KEEP_BLOCKS": "1",
        "ENABLE_NR_LAYER": "1",
        "VK_LAYER_PATH": str(installed_path(profile)),
        "VK_INSTANCE_LAYERS": "VK_LAYER_dlssnr_intel",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # DXVK logs into the folder the game was started in unless told otherwise. Through
        # the proxy its first log still goes there (_remove_dxvk_logs).
        "DXVK_LOG_PATH": str(work / "logs"),
    }
    if profile.disable_fossilize:
        values["DISABLE_VK_LAYER_VALVE_steam_fossilize_1"] = "1"
    return values


def runtime_env(profile):
    values = _clean_environment()
    values.update(layer_environment(profile))
    return values
