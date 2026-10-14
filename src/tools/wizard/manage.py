"""Install and remove NR beside a game, and switch its effect."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from wizard import config, deps, process, state


def uninstall(profile, emit=None):
    """Take NR out of the game's folder: the files it put there go, the game's own come back."""
    allowed, detail = process._install_process_guard(profile)
    if not allowed:
        return {"ok": False, "error": detail}
    destination = state.installed_path(profile)
    if not destination.exists():
        return {
            "ok": True,
            "error": None,
            "removed": False,
            "detail": "NR is not installed beside this game",
        }
    owns, detail = deps._owned_installation(profile)
    if not owns:
        return {"ok": False, "error": detail}
    # Launch options an earlier setup gave Steam would go on starting the game through a
    # wrapper whose layer is gone, and the game would not start. The bridge returns them first.
    steam = profile.root / "work/windows-wizard/steam-backup.json"
    if steam.exists():
        try:
            pending = not config._read_json(steam, limit=64 * 1024).get("restored")
        except (OSError, ValueError, AttributeError):
            pending = True
        if pending:
            return {
                "ok": False,
                "error": "Restore Steam's launch options before removing NR",
            }
    game = profile.game_exe.parent
    kept, lost = [], []
    try:
        record = deps._game_record(destination)
        placed_any = any(entry.get("placed") for entry in record["files"].values())
        for name in sorted(record["files"]):
            entry = record["files"][name]
            if entry.get("placed"):
                target = game / name
                if target.is_file() and deps._sha256(target) != entry.get("sha256"):
                    kept.append(name)
                    continue
                target.unlink(missing_ok=True)
                backup = destination / config.GAME_BACKUP / name
                if entry.get("backup") and backup.is_file():
                    os.replace(backup, target)
                    if emit:
                        emit(f"Put back the game's own {name}\n")
                elif entry.get("backup"):
                    lost.append(name)
            del record["files"][name]
            config._atomic_json(destination / config.GAME_FILES, record)
        if placed_any:
            _remove_dxvk_logs(profile, destination, record)
        if kept:
            return {
                "ok": False,
                "error": "Changed since NR put them there, so left as they are: "
                + ", ".join(kept)
                + ". The dlss-nr folder keeps the game's originals.",
                "kept": kept,
            }
        shutil.rmtree(destination)
        (profile.root / "work" / "nr_trigger").unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}
    if emit:
        emit("NR removed from the game\n")
    result = {"ok": True, "error": None, "removed": True, "trigger_exists": False}
    if lost:
        # The set-aside copies were deleted from dlss-nr by hand; Steam's file check restores them.
        result["missing_originals"] = lost
    return result


def _remove_dxvk_logs(profile, destination, record):
    """The logs NR's DXVK wrote into the game's folders.

    DXVK logs into the folder the game was started in, as <game>_<module>.log, until
    DXVK_LOG_PATH is set. The proxy sets it when DXVK loads vulkan-1.dll, after DXVK's first
    log is open, and lists that folder in START_FOLDERS. Beside the executable, the logs that
    were there before NR stay; in the other folders, those last written before it."""
    game = profile.game_exe.parent
    for name in deps._dxvk_logs(profile):
        if name not in record.get("logs_before", []):
            (game / name).unlink(missing_ok=True)
    since = record.get("installed_time")
    started = destination / config.START_FOLDERS
    if not isinstance(since, (int, float)) or not started.is_file():
        return
    for line in started.read_text(encoding="utf-8", errors="replace").splitlines():
        folder = Path(line.strip())
        # A folder that cannot be reached keeps its log, not NR.
        try:
            if not folder.is_absolute() or config._canonical(folder) == config._canonical(game):
                continue
            for name in deps._dxvk_logs(profile):
                log = folder / name
                # A file's time can be coarser than this clock: 2 s on FAT.
                if log.is_file() and not log.is_symlink() and log.stat().st_mtime >= since - 2:
                    log.unlink()
        except OSError as error:
            sys.stderr.write(f"[wizard] cannot clear a dxvk log: {error}\n")
            continue


def install(profile, emit=None):
    allowed, detail = process._install_process_guard(profile)
    if not allowed:
        return {"ok": False, "error": detail}
    checked = deps.validate(profile)
    failed = {item["name"] for item in checked["checks"] if not item["ok"]}
    if failed == {"python_packages"}:
        prepared = deps.ensure_dependencies(profile, emit=emit)
        if not prepared["ok"]:
            return prepared
        profile = config.Profile.from_dict(prepared["profile"])
        checked = deps.validate(profile)
    if not checked["ok"]:
        return checked
    commands = []
    if not checked["weights"]["ready"]:
        if emit:
            emit("Extracting logical weights from your local NVIDIA DLL\n")
        extracted = process._run(
            [
                profile.python,
                "-I",
                str(profile.root / "scripts/get_weights.py"),
                profile.dll,
                "--work-dir",
                str(profile.root / "work"),
            ],
            timeout=900,
            cwd=profile.root,
            emit=emit,
        )
        commands.append(extracted)
        # Kept for Save report: the window shows it only until the next action (issue #12).
        try:
            (profile.root / "work/logs").mkdir(parents=True, exist_ok=True)
            (profile.root / config.WEIGHTS_LOG).write_text(extracted["output"], encoding="utf-8")
        except OSError as error:
            sys.stderr.write(f"[wizard] cannot write the weights log: {error}\n")
        if (
            extracted["returncode"] != 0
            or extracted["timed_out"]
            or not deps._weights_state(profile)["ready"]
        ):
            version = deps._file_version(profile.dll)
            return {
                "ok": False,
                "error": "Weight extraction failed or did not produce 649 logical tensors, "
                f"from nvngx_dlssnr.dll version {version or 'unknown'} (tested: {config.TESTED_DLL_VERSION}). "
                "The extractor's output is in Check details and in Save report.",
                "commands": commands,
                "output": extracted["output"],
            }
    # Recheck every input and destination after extraction, before any game write.
    checked = deps.validate(profile)
    if not checked["ok"]:
        return checked
    allowed, detail = process._install_process_guard(profile)
    if not allowed:
        return {"ok": False, "error": detail}
    destination = state.installed_path(profile)
    stage = backup = None
    committed = False
    work = profile.root / "work"
    state_files = [
        state.profile_path(profile.root),
        work / "nr_settings.json",
        work / "nr_trigger",
    ]
    saved = {}
    placed = []
    try:
        saved = {path: path.read_bytes() if path.exists() else None for path in state_files}
        stage = Path(tempfile.mkdtemp(prefix=".dlss-nr-stage-", dir=destination.parent))
        if destination.exists():
            shutil.copytree(destination, stage, dirs_exist_ok=True)
        deps._copy_runtime(profile, stage, destination)
        placed = deps._place_game_files(profile, process.pe_architecture(profile.game_exe), stage)
        deps._write_proxy_environment(profile, stage, destination)
        # What the game's process writes into: DXVK's logs, and the proxy's launch record,
        # for which it makes no folder.
        (work / "logs").mkdir(parents=True, exist_ok=True)
        (work / "windows-wizard").mkdir(parents=True, exist_ok=True)
        allowed, detail = process._install_process_guard(profile)
        if not allowed:
            raise ValueError(detail)
        owns, detail = deps._owned_installation(profile)
        if not owns:
            raise ValueError(detail)
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=".dlss-nr-backup-", dir=destination.parent))
            backup.rmdir()
            destination.rename(backup)
        stage.rename(destination)
        stage = None
        committed = True
        settings = {}
        if saved[work / "nr_settings.json"]:
            settings = json.loads(saved[work / "nr_settings.json"].decode("utf-8-sig"))
            if not isinstance(settings, dict):
                raise ValueError("Settings must be a JSON object")
        settings.setdefault("render_scale", 0.4)
        settings.setdefault("min_extent", 320)
        config._atomic_json(work / "nr_settings.json", settings)
        (work / "nr_trigger").unlink(missing_ok=True)
        state.save_profile(profile)
    except (OSError, ValueError, TypeError, KeyError) as error:
        if placed:
            deps._undo_game_files(profile, destination if committed else stage, placed)
        if committed:
            shutil.rmtree(destination)
        if backup and backup.exists():
            backup.rename(destination)
            backup = None
        if committed:
            for path, content in saved.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
        return {"ok": False, "error": str(error), "commands": commands}
    finally:
        if stage and stage.exists():
            shutil.rmtree(stage)
    if backup and backup.exists():
        shutil.rmtree(backup)
    shutil.rmtree(destination / ".undo", ignore_errors=True)
    if emit:
        emit("Installation complete; the effect is off\n")
    return {
        "ok": True,
        "error": None,
        "installed": str(destination),
        "root": str(profile.root),
        "profile": profile.to_dict(),
        "commands": commands,
        "trigger_exists": False,
    }


def set_effect(profile, enabled):
    owns, detail = deps._owned_installation(profile)
    if not owns or not (state.installed_path(profile) / config.OWNER_NAME).is_file():
        return {
            "ok": False,
            "error": detail if not owns else "Install this profile before toggling its effect",
        }
    trigger = Path(state.runtime_env(profile)["NR_LAYER_TRIGGER"])
    try:
        launch = process._launch_state(profile)
        if launch.get("game_exe") and config._canonical(launch["game_exe"]) != config._canonical(
            profile.game_exe
        ):
            if process._running_executable(launch.get("pid")):
                raise ValueError(
                    "The active game uses another profile; select it before changing its effect"
                )
        if trigger.is_symlink() or (trigger.exists() and not trigger.is_file()):
            raise ValueError("The release trigger path is not an owned regular file")
        if enabled:
            trigger.parent.mkdir(parents=True, exist_ok=True)
            trigger.touch()
        else:
            trigger.unlink(missing_ok=True)
        result = {
            "ok": True,
            "error": None,
            "trigger_exists": bool(enabled),
            "trigger": str(trigger),
            "processing_confirmed": False,
            "detail": "Trigger changed; fresh processed frames must confirm processing",
        }
        if enabled and _game_running_without_nr(profile, launch):
            result.update(
                warning="game_without_nr",
                detail="The game is running without NR's layer: it was started before NR "
                "was installed, or it does not load the Vulkan loader from its own "
                "folder. Restart it; if that does not help, use Launch game.",
            )
        return result
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


def _game_running_without_nr(profile, launch):
    """Whether the selected game runs without NR's layer: a process of it that no launch names.

    One started before NR was installed is such a process, and so is a game that loads Windows'
    own vulkan-1.dll by its path, past the proxy. Switching the effect on changes nothing in
    it. One process list, taken only when the effect is switched on, not by the status check
    that runs every 5 s."""
    target = config._canonical(profile.game_exe)
    if (
        launch.get("game_exe")
        and config._canonical(launch["game_exe"]) == target
        and launch.get("exit_code") is None
    ):
        running = process._running_executable(launch.get("pid"))
        if running and config._canonical(running) == target:
            return False
    try:
        processes = process._process_snapshot()
    except ValueError:
        return False
    return any(
        item.get("ExecutablePath") and config._canonical(item["ExecutablePath"]) == target
        for item in processes
    )
