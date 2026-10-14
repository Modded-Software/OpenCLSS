"""The daemon-log observer and the saved report."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from wizard import config, deps, process
from wizard.state import runtime_env

_LOG_STATES = {}
_LOG_LOCK = threading.Lock()


def _new_log_state(complete=True):
    return {
        "offset": 0,
        "pending": b"",
        "identity": None,
        "prefix": None,
        "processed": 0,
        "rejected": 0,
        "errors": 0,
        "error_tail": deque(maxlen=30),
        "last_frame": None,
        "network_shape": None,
        "half_probe": None,
        "allocator": False,
        "model_ready": None,
        "session": 0,
        "counts_complete": complete,
    }


def _parse_log_line(state, line):
    if line.startswith("half rounding:"):
        previous_session = state["session"]
        for name, value in _new_log_state().items():
            if name not in ("offset", "pending", "identity", "prefix"):
                state[name] = value
        state["session"] = previous_session + 1
        state["half_probe"] = {
            "checked": "NOT CHECKED" not in line,
            "ok": None,
            "detail": line,
        }
    if "half_round uses" in line and state["half_probe"]:
        state["half_probe"]["detail"] += "\n" + line
    if line.startswith("model ready"):
        if state["model_ready"]:
            previous_session = state["session"]
            for name, value in _new_log_state().items():
                if name not in ("offset", "pending", "identity", "prefix"):
                    state[name] = value
            state["session"] = previous_session + 1
        state["model_ready"] = line
        if (
            state["half_probe"]
            and state["half_probe"]["checked"]
            and state["half_probe"]["ok"] is None
        ):
            state["half_probe"]["ok"] = True
    if "keeping NumPy's large blocks" in line:
        state["allocator"] = True
    if line.startswith("frame rejected/failed") or "unsupported VkFormat" in line:
        state["rejected"] += 1
    if any(
        word in line
        for word in (
            "frame rejected/failed",
            "GPU lost",
            "Traceback",
            "FAIL:",
            "ERROR",
            "unsupported VkFormat",
        )
    ):
        state["errors"] += 1
        state["error_tail"].append(line[:500])
        if line.startswith("FAIL:") and state["half_probe"]:
            state["half_probe"]["ok"] = False
    match = re.match(r"^(\d+)x(\d+)\s+\S+\s+in\s+(\d+(?:\.\d+)?)s\s+change\s+(\S+)", line)
    if match:
        state["processed"] += 1
        network = re.search(r"\bnetwork (\d+)x(\d+)", line)
        state["network_shape"] = [int(value) for value in network.groups()] if network else None
        state["last_frame"] = {
            "width": int(match[1]),
            "height": int(match[2]),
            "seconds": float(match[3]),
            "change": match[4],
            "line": line[:1000],
        }
        if match[4].lower() in ("nan", "inf", "-inf"):
            state["errors"] += 1
            state["error_tail"].append("Non-finite image change: " + line[:400])


def status(profile):
    log = Path(runtime_env(profile)["NR_LAYER_LOG"])
    key = config._canonical(log)
    observer = profile.root / "work/windows-wizard/status-observer.json"
    try:
        launch = process._launch_state(profile)
    except (OSError, ValueError):
        launch = {}
    matching_launch = bool(
        launch.get("root")
        and launch.get("game_exe")
        and config._canonical(launch["root"]) == config._canonical(profile.root)
        and config._canonical(launch["game_exe"]) == config._canonical(profile.game_exe)
    )
    context = {
        "root": config._canonical(profile.root),
        "game_exe": config._canonical(profile.game_exe),
        "pid": launch.get("pid") if matching_launch else None,
        "started_utc": launch.get("started_utc") if matching_launch else None,
        "start_offset": launch.get("daemon_start_bytes", 0) if matching_launch else 0,
    }
    if not isinstance(context["start_offset"], int) or context["start_offset"] < 0:
        context["start_offset"] = 0
    context_key = hashlib.sha256(json.dumps(context, sort_keys=True).encode("utf-8")).hexdigest()
    with _LOG_LOCK:
        state = _LOG_STATES.get(key)
        if state is None:
            try:
                saved = config._read_json(observer, limit=256 * 1024)
                if (
                    saved.get("schema_version") == config.SCHEMA_VERSION
                    and saved.get("context_key") == context_key
                ):
                    state = saved["state"]
                    state["pending"] = base64.b64decode(state["pending"])
                    state["prefix"] = (
                        bytes.fromhex(state["prefix"]) if state["prefix"] is not None else None
                    )
                    state["identity"] = tuple(state["identity"]) if state["identity"] else None
                    state["error_tail"] = deque(state["error_tail"], maxlen=30)
            except (OSError, ValueError, TypeError, KeyError):
                state = None
        if state is None or state.get("context_key") != context_key:
            state = _new_log_state()
            state["offset"] = context["start_offset"]
            state["context_key"] = context_key
        _LOG_STATES[key] = state
        initial = state["identity"] is None
        fresh_processed = 0
        mtime = None
        try:
            stat = log.stat()
            mtime = stat.st_mtime
            with log.open("rb") as handle:
                prefix = handle.read(64)
                identity = (stat.st_dev, stat.st_ino)
                # A short file's first 64 bytes legitimately grow. Only a changed
                # existing prefix, replaced file or truncation starts a new log.
                prefix_changed = state["prefix"] is not None and not prefix.startswith(
                    state["prefix"]
                )
                if (
                    (state["identity"] is not None and identity != state["identity"])
                    or stat.st_size < state["offset"]
                    or prefix_changed
                ):
                    state = _new_log_state(complete=stat.st_size <= config.MAX_LOG_READ)
                    state["context_key"] = context_key
                    _LOG_STATES[key] = state
                    initial = True
                offset = state["offset"]
                if stat.st_size - offset > config.MAX_LOG_READ:
                    offset = stat.st_size - config.MAX_LOG_READ
                    state["counts_complete"] = False
                    state["pending"] = b""
                handle.seek(offset)
                content = handle.read(min(config.MAX_LOG_READ, max(0, stat.st_size - offset)))
                consumed = handle.tell()
            if offset and offset != state["offset"]:
                content = content.split(b"\n", 1)[-1]
            before = state["processed"]
            content = state["pending"] + content
            lines = content.split(b"\n")
            state["pending"] = lines.pop()
            for line in lines:
                _parse_log_line(state, line.decode("utf-8", errors="replace").rstrip("\r"))
            # First observation after an authenticated launch may contain its
            # first frames. An unassociated historical log cannot confirm them.
            fresh_processed = (
                max(0, state["processed"] - before) if not initial or matching_launch else 0
            )
            state.update(
                offset=consumed,
                identity=identity,
                prefix=prefix,
                context_key=context_key,
            )
        except FileNotFoundError:
            state = _new_log_state()
            state["context_key"] = context_key
            _LOG_STATES[key] = state
        except OSError as error:
            return {"ok": False, "error": str(error), "log_path": str(log)}
        result = {
            name: value
            for name, value in state.items()
            if name
            not in (
                "offset",
                "pending",
                "identity",
                "prefix",
                "error_tail",
                "context_key",
            )
        }
        result.update(
            ok=True,
            error=None,
            error_tail=list(state["error_tail"]),
            trigger_exists=Path(runtime_env(profile)["NR_LAYER_TRIGGER"]).is_file(),
            log_path=str(log),
            log_mtime=mtime,
            fresh_processed=fresh_processed,
            processing_observed=state["processed"] > 0,
            game_connected=None,
        )
        persisted = dict(
            state,
            pending=base64.b64encode(state["pending"]).decode("ascii"),
            prefix=state["prefix"].hex() if state["prefix"] is not None else None,
            identity=list(state["identity"]) if state["identity"] else None,
            error_tail=list(state["error_tail"]),
        )
        try:
            config._atomic_json(
                observer,
                {
                    "schema_version": config.SCHEMA_VERSION,
                    "context_key": context_key,
                    "state": persisted,
                },
            )
        except OSError as error:
            result["observer_error"] = str(error)
    active = False
    process_error = None
    if matching_launch and isinstance(launch.get("pid"), int) and launch.get("exit_code") is None:
        try:
            running = process._running_executable(launch["pid"])
            active = bool(running) and config._canonical(running) == config._canonical(
                profile.game_exe
            )
        except (OSError, ValueError):
            process_error = "The launch PID could not be verified"
    result.update(
        launch=context,
        active_game=active,
        fresh_frames=bool(
            active and fresh_processed > 0 and mtime is not None and time.time() - mtime < 10
        ),
    )
    if process_error:
        result["process_observation_error"] = process_error
    return result


def _system_metadata():
    values = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "gpu": [],
    }
    if os.name != "nt":
        return values
    script = """
$null = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$ErrorActionPreference = 'Stop'
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture
$gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,PNPDeviceID)
@{windows=$os;gpu=$gpu} | ConvertTo-Json -Depth 5 -Compress
"""
    command = "powershell.exe"
    system_root = os.environ.get("SystemRoot")
    if system_root:
        command = str(Path(system_root) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    got = process._run([command, "-NoProfile", "-NonInteractive", "-Command", script], timeout=8)
    try:
        if got["returncode"] == 0:
            values.update(json.loads(got["stdout"]))
        else:
            values["hardware_query_error"] = got["output"]
    except ValueError:
        values["hardware_query_error"] = got["output"]
    return values


def export_report(profile, destination):
    destination = Path(destination)
    if destination.is_dir():
        destination /= "nr-windows-report-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + ".json"
    if destination.suffix.lower() != ".json":
        return {"ok": False, "error": "Report destination must be a JSON file"}
    try:
        observation = status(profile)
        log = Path(runtime_env(profile)["NR_LAYER_LOG"])
        tail = ""
        if log.is_file():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - config.MAX_LOG_TAIL))
                tail = handle.read(config.MAX_LOG_TAIL).decode("utf-8", errors="replace")
        metadata = None
        metadata_error = None
        path = profile.root / "release-metadata.json"
        if path.is_file():
            try:
                metadata = config._read_json(path, limit=64 * 1024)
            except (OSError, ValueError) as error:
                metadata_error = str(error)
        try:
            settings = config._read_json(profile.root / "work/nr_settings.json")
            settings_error = None
        except (OSError, ValueError) as error:
            settings = None
            settings_error = str(error)
        # What Check says, so a report of a failed installation names the item that failed; the
        # DLL's version, and what the weight extractor said last (issue #12).
        try:
            checks = deps.validate(profile)["checks"]
        except (OSError, ValueError) as error:
            checks = [{"name": "validate", "ok": False, "detail": str(error)}]
        weights_log = profile.root / config.WEIGHTS_LOG
        weights_tail = (
            weights_log.read_bytes()[-config.MAX_LOG_TAIL :].decode("utf-8", errors="replace")
            if weights_log.is_file()
            else ""
        )
        report = {
            "schema_version": config.SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "release": metadata,
            "release_metadata_error": metadata_error,
            "root": str(profile.root),
            "profile": profile.to_dict(),
            "system": _system_metadata(),
            "python": process._python_info(profile.python),
            "checks": checks,
            "dll": deps._dll_info(profile.dll),
            "weights_log_tail": weights_tail,
            "settings": settings,
            "settings_error": settings_error,
            "status": observation,
            "daemon_log_tail": tail,
            "note": "Daemon timings and observed frame counts are not game FPS. No DLL or weights are included.",
        }
        config._atomic_json(destination, report)
        return {"ok": True, "error": None, "path": str(destination)}
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}
