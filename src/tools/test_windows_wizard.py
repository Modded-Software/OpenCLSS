"""CPU-only release fixtures; no game, Steam, vendor weights or GPU execution."""

import ast
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from wizard import config, deps, manage, process, state, status


def write_pe(path, x64=True, suffix=b""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytearray(90)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, 0x8664 if x64 else 0x14C)
    struct.pack_into("<H", data, 88, 0x20B if x64 else 0x10B)
    path.write_bytes(data + suffix)
    return path


def write_weights(path):
    """Format-only zero-sized tensors; deliberately contain no real weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {f"fixture_{n}": {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]} for n in range(649)}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def python_info(**changes):
    result = {
        "probe_ok": True,
        "platform": "win32",
        "os_name": "nt",
        "bits": 64,
        "abi_platform": "win-amd64",
        "native_apis": True,
        "version": "3.12.10",
        "packages": {"numpy": "fixture", "safetensors": "fixture"},
        "package_errors": {},
        "output": "",
    }
    result.update(changes)
    return result


class WizardFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nr wizard (fixture) ")
        self.addCleanup(self.temporary.cleanup)
        # Resolved, as the wizard resolves what it is given: a TEMP in 8.3 form
        # (C:\Users\NAME~1\...) otherwise compares unequal to the same file's long name.
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "release (candidate)"
        self.root.mkdir()
        for relative in config.RUNTIME_MODULES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# release fixture\n", encoding="utf-8")
        for directory in config.SOURCE_DIRS:
            (self.root / "src" / directory).mkdir(parents=True, exist_ok=True)
        write_pe(self.root / "nr_layer.dll")
        write_pe(self.root / "nr_layer32.dll", x64=False)
        write_pe(self.root / "nr_vulkan_proxy.dll", suffix=b"proxy x64")
        write_pe(self.root / "nr_vulkan_proxy32.dll", x64=False, suffix=b"proxy x32")
        for directory, x64 in (("x64", True), ("x32", False)):
            for name in config.DXVK_FILES:
                write_pe(
                    self.root / "dxvk" / directory / name,
                    x64=x64,
                    suffix=f"DXVK {directory} {name}".encode(),
                )
        for name in config.LIBRARIES:
            write_pe(self.root / "work" / name)
        for name in config.SHADERS:
            (self.root / "work" / name).write_bytes(b"SPIR-V fixture")
        config._atomic_json(
            self.root / "VkLayer_dlss_nr.json",
            {
                "file_format_version": "1.0.0",
                "layer": {
                    "name": "VK_LAYER_dlssnr_intel",
                    "type": "GLOBAL",
                    "library_path": "nr_layer.dll",
                    "api_version": "1.3.0",
                },
            },
        )
        self.game = write_pe(self.base / "game (test)" / "selected.exe")
        self.python = write_pe(self.base / "Python (selected)" / "python.exe")
        self.profile = config.Profile(self.root, self.python, self.game)
        self.weights = self.root / "work/mlxw/dlssnr-logical.safetensors"
        write_weights(self.weights)
        self.native = patch.object(process, "_python_info", return_value=python_info()).start()
        self.deps = patch.object(
            deps, "_load_dependencies", return_value={"ok": True, "failures": {}}
        ).start()
        self.processes = patch.object(process, "_process_snapshot", return_value=[]).start()
        self.running = patch.object(process, "_running_executable", return_value=None).start()
        self.addCleanup(patch.stopall)
        status._LOG_STATES.clear()
        self.addCleanup(status._LOG_STATES.clear)

    def checks(self, profile=None):
        return {item["name"]: item for item in deps.validate(profile or self.profile)["checks"]}

    def own(self, profile=None):
        profile = profile or self.profile
        directory = state.installed_path(profile)
        directory.mkdir(parents=True, exist_ok=True)
        config._atomic_json(
            directory / config.OWNER_NAME,
            {"schema_version": 1, "root": str(profile.root)},
        )
        return directory

    def launch_state(self, profile=None, pid=123, offset=0, started="fixture-session"):
        profile = profile or self.profile
        config._atomic_json(
            profile.root / "work/windows-wizard/launch-state.json",
            {
                "root": str(profile.root),
                "game_exe": str(profile.game_exe),
                "pid": pid,
                "daemon_start_bytes": offset,
                "started_utc": started,
                "exit_code": None,
            },
        )
        self.processes.return_value = [{"ProcessId": pid, "ExecutablePath": str(profile.game_exe)}]
        self.running.side_effect = lambda value, pid=pid, path=str(profile.game_exe): (
            path if value == pid else None
        )

    def test_profile_defaults_round_trip_and_paths(self):
        minimal = config.Profile.from_dict(
            {
                "root": str(self.root),
                "python": str(self.python),
                "game_exe": str(self.game),
            }
        )
        assert minimal.api == "vulkan"
        assert minimal.game_args == []
        assert minimal.dll is None
        assert config.Profile.from_dict(minimal.to_dict()) == minimal
        assert state.save_profile(minimal)["ok"]
        assert state.load_profile(self.root) == minimal

    def test_steam_blank_app_id_allows_autodetection_and_invalid_id_is_rejected(self):
        for app_id, expected in (
            ("", True),
            ("379720", True),
            ("DOOM", False),
            ("379720 extra", False),
        ):
            with self.subTest(app_id=app_id):
                profile = replace(self.profile, launch_mode="steam", steam_app_id=app_id)
                result = deps.validate(profile)
                assert result["ok"] == expected
                assert self.checks(profile)["launch_mode"]["ok"] == expected
        profile = replace(self.profile, launch_mode="steam", steam_app_id="")
        result = manage.install(profile)
        assert result["ok"], result
        assert state.load_profile(self.root).steam_app_id == ""

    def test_runtime_env_same_python_unique_pipe_and_clean_inherited_layers(self):
        with patch.dict(
            os.environ,
            {
                "NR_ROOT": "stale",
                "NR_PYTHON": "stale",
                "XMX_TEST": "1",
                "VK_LAYER_PATH": "stale",
                "VK_DEVICE_LAYERS": "stale",
                "DISABLE_NR_LAYER": "1",
                "SteamAppId": "2280",
            },
        ):
            env = state.runtime_env(self.profile)
        assert env["NR_PYTHON"] == str(self.python)
        assert env["NR_ROOT"] == str(self.root)
        assert env["NR_DAEMON"] == str(self.root / "src/layer/nr_daemon.py")
        assert env["NR_SETTINGS"] == str(self.root / "work/nr_settings.json")
        assert env["NR_LAYER_TRIGGER"] == str(self.root / "work/nr_trigger")
        assert {key.upper(): value for key, value in env.items()}["STEAMAPPID"] == "2280"
        for name in ("XMX_TEST", "VK_DEVICE_LAYERS", "DISABLE_NR_LAYER"):
            assert name not in env
        assert env["VK_LAYER_PATH"] == str(self.game.parent / "dlss-nr")
        assert (
            env["NR_LAYER_SOCKET"]
            != state.runtime_env(replace(self.profile, root=self.base / "other"))["NR_LAYER_SOCKET"]
        )
        assert env["NR_LAYER_SOCKET"] == state.runtime_env(self.profile)["NR_LAYER_SOCKET"]
        assert "DISABLE_VK_LAYER_VALVE_steam_overlay_1" not in env
        assert "DISABLE_VK_LAYER_VALVE_steam_fossilize_1" not in state.runtime_env(
            replace(self.profile, disable_fossilize=False)
        )

    def test_32bit_game_accepted_and_other_machines_and_32bit_python_rejected(self):
        write_pe(self.game, x64=False)
        assert self.checks()["game_architecture"]["ok"]
        arm64 = bytearray(write_pe(self.game).read_bytes())
        struct.pack_into("<H", arm64, 68, 0xAA64)
        self.game.write_bytes(bytes(arm64))
        assert not self.checks()["game_architecture"]["ok"]
        write_pe(self.game)
        self.native.return_value = python_info(bits=32, abi_platform="win32")
        assert not self.checks()["python_native_x64"]["ok"]
        assert not manage.install(self.profile)["ok"]
        assert not state.installed_path(self.profile).exists()

    def test_mingw_abi_or_missing_native_apis_are_rejected(self):
        for change in ({"abi_platform": "mingw_x86_64"}, {"native_apis": False}):
            with self.subTest(change=change):
                self.native.return_value = python_info(**change)
                assert not self.checks()["python_native_x64"]["ok"]

    def test_discovery_excludes_mingw_even_when_inherited_nr_python(self):
        mingw = write_pe(self.base / "mingw Python" / "python.exe")
        self.native.side_effect = lambda candidate, **kwargs: (
            python_info(abi_platform="mingw_x86_64") if Path(candidate) == mingw else python_info()
        )
        with (
            patch.dict(os.environ, {"NR_PYTHON": str(mingw)}),
            patch.object(sys, "executable", str(self.python)),
            patch.object(shutil, "which", return_value=None),
        ):
            found = process.discover_python()
        assert str(self.python) in found
        assert str(mingw) not in found

    def test_every_runtime_shader_and_allocator_must_be_regular_file(self):
        for relative in (
            "work/gemm_staged.spv",
            "work/half_probe.spv",
            "work/libnr_alloc.dll",
        ):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.unlink()
                path.mkdir()
                assert not self.checks()["runtime_files"]["ok"]
                assert not manage.install(self.profile)["ok"]
                assert not state.installed_path(self.profile).exists()
                path.rmdir()
                path.write_bytes(original)

    def test_loader_or_runtime_dependency_failure_is_preflight_failure(self):
        self.deps.return_value = {
            "ok": False,
            "failures": {"vulkan-1.dll": "missing loader"},
        }
        result = manage.install(self.profile)
        assert not result["ok"]
        assert not state.installed_path(self.profile).exists()
        assert "missing loader" in self.checks()["runtime_dependencies"]["detail"]

    def test_source_directory_missing_is_rejected_before_game_staging(self):
        (self.root / "src/bench").rmdir()
        assert not self.checks()["runtime_files"]["ok"]
        with patch.object(
            tempfile,
            "mkdtemp",
            side_effect=AssertionError("No game writes after failed preflight"),
        ):
            assert not manage.install(self.profile)["ok"]

    def test_dxvk_comes_from_the_release_for_the_games_architecture(self):
        profile = replace(self.profile, api="dxvk")
        assert self.checks(profile)["dxvk_files"]["ok"]
        write_pe(self.root / "dxvk/x64/dxgi.dll", x64=False)
        assert not self.checks(profile)["dxvk_files"]["ok"]
        write_pe(self.game, x64=False)
        assert self.checks(profile)["dxvk_files"]["ok"]

    def test_dxvk_install_sets_the_games_files_aside_and_remove_puts_them_back(self):
        profile = replace(self.profile, api="dxvk")
        game = self.game.parent
        (game / "dxgi.dll").write_bytes(b"the game's own proxy")
        shutil.copy2(self.root / "dxvk/x64/d3d9.dll", game / "d3d9.dll")  # someone's identical copy
        (game / f"{self.game.stem}_d3d9.log").write_text("that DXVK's own log")
        assert "dxgi.dll" in self.checks(profile)["game_files"]["detail"]
        result = manage.install(profile)
        assert result["ok"], result
        for name in config.DXVK_FILES:
            assert (game / name).read_bytes() == (self.root / "dxvk/x64" / name).read_bytes()
        installed = state.installed_path(profile)
        assert (installed / config.GAME_BACKUP / "dxgi.dll").read_bytes() == b"the game's own proxy"
        manifest = json.loads((installed / "VkLayer_dlss_nr.json").read_text(encoding="utf-8"))
        assert manifest["layer"]["library_arch"] == "64"
        assert Path(state.runtime_env(profile)["DXVK_LOG_PATH"]) == self.root / "work/logs"
        # Installed again, the game's original is still the one set aside, once.
        assert manage.install(profile)["ok"]
        assert (installed / config.GAME_BACKUP / "dxgi.dll").read_bytes() == b"the game's own proxy"
        assert not (installed / ".undo").exists()
        (game / f"{self.game.stem}_dxgi.log").write_text("a DXVK log from a launch outside NR")
        removed = manage.uninstall(profile)
        assert removed["ok"], removed
        assert (game / "dxgi.dll").read_bytes() == b"the game's own proxy"
        assert (game / "d3d9.dll").read_bytes() == (self.root / "dxvk/x64/d3d9.dll").read_bytes()
        for name in (
            "d3d8.dll",
            "d3d10core.dll",
            "d3d11.dll",
            f"{self.game.stem}_dxgi.log",
        ):
            assert not (game / name).exists(), name
        assert not installed.exists()
        assert (game / f"{self.game.stem}_d3d9.log").read_text() == "that DXVK's own log"
        assert sorted(path.name for path in game.iterdir()) == sorted(
            ["d3d9.dll", "dxgi.dll", self.game.name, f"{self.game.stem}_d3d9.log"]
        )

    def test_dxvk_placement_failure_puts_the_games_files_back(self):
        profile = replace(self.profile, api="dxvk")
        game = self.game.parent
        (game / "dxgi.dll").write_bytes(b"original")
        copy = shutil.copy2

        def failing(source, target, *args, **kwargs):
            if Path(source).name == "dxgi.dll":
                raise OSError("disk full")
            return copy(source, target, *args, **kwargs)

        with patch.object(shutil, "copy2", side_effect=failing):
            assert not manage.install(profile)["ok"]
        assert sorted(path.name for path in game.iterdir()) == [
            "dxgi.dll",
            self.game.name,
        ]
        assert (game / "dxgi.dll").read_bytes() == b"original"

    def test_late_failure_after_dxvk_commit_puts_the_games_files_back(self):
        profile = replace(self.profile, api="dxvk")
        game = self.game.parent
        (game / "dxgi.dll").write_bytes(b"original")
        with patch.object(state, "save_profile", side_effect=OSError("profile write failed")):
            assert not manage.install(profile)["ok"]
        assert sorted(path.name for path in game.iterdir()) == [
            "dxgi.dll",
            self.game.name,
        ]
        assert (game / "dxgi.dll").read_bytes() == b"original"

    def test_32bit_game_gets_the_32bit_layer_and_dxvk(self):
        write_pe(self.game, x64=False)
        profile = replace(self.profile, api="dxvk")
        assert manage.install(profile)["ok"]
        installed = state.installed_path(profile)
        manifest = json.loads((installed / "VkLayer_dlss_nr.json").read_text(encoding="utf-8"))
        assert manifest["layer"]["library_arch"] == "32"
        assert Path(manifest["layer"]["library_path"]) == installed / "nr_layer32.dll"
        assert process.pe_architecture(installed / "nr_layer32.dll") == "x86"
        assert (self.game.parent / "d3d9.dll").read_bytes() == (
            self.root / "dxvk/x32/d3d9.dll"
        ).read_bytes()
        assert (self.game.parent / config.PROXY_NAME).read_bytes() == (
            self.root / "nr_vulkan_proxy32.dll"
        ).read_bytes()

    def test_every_game_gets_the_proxy_and_its_environment(self):
        game = self.game.parent
        result = manage.install(self.profile)
        assert result["ok"], result
        assert (game / config.PROXY_NAME).read_bytes() == (
            self.root / "nr_vulkan_proxy.dll"
        ).read_bytes()
        installed = state.installed_path(self.profile)
        lines = (installed / config.ENV_FILE).read_text(encoding="utf-8").splitlines()
        values = dict(
            line.split("=", 1) for line in lines if "=" in line and not line.startswith("#")
        )
        assert Path(values["NR_GAME_EXE"]) == self.game
        assert (
            Path(values["NR_LAUNCH_STATE"]) == self.root / "work/windows-wizard/launch-state.json"
        )
        # The proxy makes no folder for its record: without one, status never saw the game.
        assert (self.root / "work/windows-wizard").is_dir()
        assert "NR_REAL_VULKAN" not in values
        for name, value in state.layer_environment(self.profile).items():
            assert values[name] == value, name
        assert manage.uninstall(self.profile)["ok"]
        assert sorted(path.name for path in game.iterdir()) == [self.game.name]

    def test_remove_takes_dxvks_first_log_from_the_folders_the_game_started_in(self):
        # DXVK opens its first log before it loads the proxy, so in the folder the game was
        # started in: Steam starts Mortal Kombat 11 in its root, above the executable.
        profile = replace(self.profile, api="dxvk")
        game = self.game.parent
        started = self.base / "started (here)"
        started.mkdir()
        old = started / f"{self.game.stem}_d3d9.log"
        old.write_text("a log from before NR")
        own = game / f"{self.game.stem}_d3d11.log"
        own.write_text("beside the game before NR")
        assert manage.install(profile)["ok"]
        installed = state.installed_path(profile)
        since = json.loads((installed / config.GAME_FILES).read_text(encoding="utf-8"))[
            "installed_time"
        ]
        os.utime(old, (since - 60, since - 60))
        (started / f"{self.game.stem}_dxgi.log").write_text("NR's DXVK, before DXVK_LOG_PATH")
        (started / "notes.txt").write_text("not DXVK's")
        own.write_text("beside the game before NR, written again")
        (installed / config.START_FOLDERS).write_text(
            "\n".join(
                [
                    str(started),
                    str(started),
                    str(game),
                    str(self.base / "gone"),
                    "relative",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        removed = manage.uninstall(profile)
        assert removed["ok"], removed
        assert sorted(path.name for path in started.iterdir()) == sorted([old.name, "notes.txt"])
        assert sorted(path.name for path in game.iterdir()) == sorted([own.name, self.game.name])

    def test_a_games_own_vulkan_loader_is_set_aside_and_used(self):
        game = self.game.parent
        (game / config.PROXY_NAME).write_bytes(b"the game's own Vulkan loader")
        assert config.PROXY_NAME in self.checks()["game_files"]["detail"]
        assert manage.install(self.profile)["ok"]
        installed = state.installed_path(self.profile)
        aside = installed / config.GAME_BACKUP / config.PROXY_NAME
        assert aside.read_bytes() == b"the game's own Vulkan loader"
        assert (
            "NR_REAL_VULKAN=" + str(aside)
            in (installed / config.ENV_FILE).read_text(encoding="utf-8").splitlines()
        )
        assert manage.uninstall(self.profile)["ok"]
        assert (game / config.PROXY_NAME).read_bytes() == b"the game's own Vulkan loader"

    def test_the_proxy_must_match_the_games_architecture(self):
        assert self.checks()["runtime_architecture"]["ok"]
        write_pe(self.root / "nr_vulkan_proxy.dll", x64=False)
        assert not self.checks()["runtime_architecture"]["ok"]

    def test_changing_the_api_after_dxvk_needs_remove_first(self):
        dxvk = replace(self.profile, api="dxvk")
        assert manage.install(dxvk)["ok"]
        assert not self.checks(self.profile)["game_files"]["ok"]
        assert not manage.install(self.profile)["ok"]
        assert manage.uninstall(dxvk)["ok"]
        assert manage.install(self.profile)["ok"]
        assert not (self.game.parent / "dxgi.dll").exists()

    def test_remove_leaves_files_changed_since_and_keeps_the_originals(self):
        profile = replace(self.profile, api="dxvk")
        (self.game.parent / "d3d11.dll").write_bytes(b"the game's d3d11")
        assert manage.install(profile)["ok"]
        (self.game.parent / "d3d11.dll").write_bytes(b"a game update")
        result = manage.uninstall(profile)
        assert not result["ok"]
        assert result["kept"] == ["d3d11.dll"]
        assert (self.game.parent / "d3d11.dll").read_bytes() == b"a game update"
        assert not (self.game.parent / "dxgi.dll").exists()
        backup = state.installed_path(profile) / config.GAME_BACKUP / "d3d11.dll"
        assert backup.read_bytes() == b"the game's d3d11"

    def test_remove_waits_for_steams_launch_options_to_be_restored(self):
        profile = replace(self.profile, api="dxvk")
        assert manage.install(profile)["ok"]
        steam = self.root / "work/windows-wizard/steam-backup.json"
        config._atomic_json(steam, {"restored": False})  # the same, unchanged profile
        assert not manage.uninstall(profile)["ok"]
        assert (self.game.parent / "dxgi.dll").exists()
        config._atomic_json(steam, {"restored": True})
        assert manage.uninstall(profile)["ok"]
        assert not (self.game.parent / "dxgi.dll").exists()

    def test_remove_nr_returns_steams_launch_options_first(self):
        import windows_launch as launch
        import windows_wizard as bridge

        with (
            patch.object(
                launch,
                "restore_steam",
                return_value={"ok": False, "error": "A Steam game is running"},
            ),
            patch.object(manage, "uninstall") as uninstall,
        ):
            result = bridge.remove_nr(self.profile, None)
        assert result["error"] == "A Steam game is running"
        uninstall.assert_not_called()
        with (
            patch.object(
                launch, "restore_steam", return_value={"ok": True, "changed": True}
            ) as restore,
            patch.object(
                manage, "uninstall", return_value={"ok": True, "removed": True}
            ) as uninstall,
        ):
            result = bridge.remove_nr(self.profile, None)
        assert result["ok"]
        assert result["steam_restored"]
        assert restore.call_args.args[0] == self.profile
        uninstall.assert_called_once()
        with (
            patch.object(
                launch,
                "restore_steam",
                return_value={"ok": True, "already_restored": True, "changed": False},
            ),
            patch.object(manage, "uninstall", return_value={"ok": True, "removed": True}),
        ):
            assert "steam_restored" not in bridge.remove_nr(self.profile, None)

    def test_enabling_warns_when_the_game_runs_without_nr(self):
        self.own()
        # Started before NR was installed: no launch record names it, the game runs.
        self.processes.return_value = [{"ProcessId": 77, "ExecutablePath": str(self.game)}]
        result = manage.set_effect(self.profile, True)
        assert result["ok"]
        assert result.get("warning") == "game_without_nr"
        assert "warning" not in manage.set_effect(self.profile, False)
        # Started through NR, it is not warned about; nor is a game that is not running.
        self.launch_state(pid=77)
        assert "warning" not in manage.set_effect(self.profile, True)
        self.running.side_effect = None
        self.processes.return_value = []
        assert "warning" not in manage.set_effect(self.profile, True)

    def test_remove_refuses_while_the_game_runs_and_without_an_installation(self):
        profile = replace(self.profile, api="dxvk")
        result = manage.uninstall(profile)
        assert result["ok"]
        assert not result["removed"]
        assert manage.install(profile)["ok"]
        self.launch_state()
        assert not manage.uninstall(profile)["ok"]
        assert (self.game.parent / "dxgi.dll").exists()
        assert state.installed_path(profile).exists()

    def test_foreign_existing_installation_is_preserved(self):
        destination = state.installed_path(self.profile)
        destination.mkdir()
        sentinel = destination / "foreign.txt"
        sentinel.write_bytes(b"do not change")
        before = sentinel.read_bytes()
        assert not manage.install(self.profile)["ok"]
        assert sentinel.read_bytes() == before
        assert list(destination.iterdir()) == [sentinel]
        config._atomic_json(
            destination / config.OWNER_NAME,
            {"schema_version": 1, "root": str(self.base / "foreign")},
        )
        assert not manage.install(self.profile)["ok"]
        assert sentinel.read_bytes() == before

    def test_successful_install_spaces_parentheses_defaults_and_no_vendor_copy(self):
        trigger = self.root / "work/nr_trigger"
        trigger.touch()
        config._atomic_json(
            self.root / "work/nr_settings.json",
            {"render_scale": 0.8, "min_extent": 640, "keep_me": 7},
        )
        result = manage.install(self.profile)
        assert result["ok"], result
        saved = state.load_profile(self.root)
        assert saved.python == self.python
        assert saved.game_exe == self.game
        settings = config._read_json(self.root / "work/nr_settings.json")
        assert settings == {"render_scale": 0.8, "min_extent": 640, "keep_me": 7}
        assert not trigger.exists()
        destination = state.installed_path(self.profile)
        manifest = config._read_json(destination / "VkLayer_dlss_nr.json")
        assert manifest["layer"]["library_path"] == str(destination / "nr_layer.dll")
        assert manifest["layer"]["library_arch"] == "64"
        assert not any(destination.rglob("*.safetensors"))
        assert not any(destination.rglob("nvngx_dlssnr.dll"))
        assert self.weights.is_file()

    def test_first_install_sets_candidate_defaults_only_when_missing(self):
        result = manage.install(self.profile)
        assert result["ok"], result
        assert config._read_json(self.root / "work/nr_settings.json") == {
            "render_scale": 0.4,
            "min_extent": 320,
        }

    def test_control_schema_and_daemon_defaults_come_from_linux_catalogue(self):
        catalogue = state._runtime_knobs()
        assert [knob["name"] for knob in catalogue] == [
            "render_scale",
            "min_extent",
            "profile",
            "intensity",
            "detail_strength",
            "colour_strength",
            "temporal",
            "hold",
            "release",
            "cut_limit",
        ]
        names = {knob["name"]: knob for knob in catalogue}
        assert names["profile"]["choices"] == [
            "standard",
            "natural",
            "cinematic",
            "neutral",
        ]
        assert (names["min_extent"]["high"], names["min_extent"]["runtime_high"]) == (
            320,
            4096,
        )
        assert (names["release"]["high"], names["release"]["runtime_high"]) == (64, 255)
        daemon_source = (Path(__file__).resolve().parents[1] / "layer/nr_daemon.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(daemon_source)
        actual = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                for keyword in node.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        actual[str(node.args[0].value).removeprefix("--").replace("-", "_")] = (
                            keyword.value.value
                        )
        assert {knob["name"]: knob["default"] for knob in catalogue} == {
            knob["name"]: actual[knob["name"]] for knob in catalogue
        }

    def test_reading_control_defaults_needs_no_game_profile_or_processes_and_writes_nothing(
        self,
    ):
        with (
            patch.object(
                process,
                "_process_snapshot",
                side_effect=AssertionError("Controls do not inspect games"),
            ),
            patch.object(
                process,
                "_running_executable",
                side_effect=AssertionError("Controls do not inspect games"),
            ),
        ):
            result = state.get_settings(self.root)
        assert result["ok"]
        assert result["settings"] == result["defaults"]
        assert result["explicit_keys"] == []
        assert not state.profile_path(self.root).exists()
        assert not (self.root / "work/nr_settings.json").exists()
        assert not (self.root / "work/nr_trigger").exists()

    def test_partial_settings_save_merges_latest_values_and_preserves_trigger_and_profile(
        self,
    ):
        state.save_profile(self.profile)
        profile_before = state.profile_path(self.root).read_bytes()
        trigger = self.root / "work/nr_trigger"
        trigger.write_bytes(b"existing effect")
        path = self.root / "work/nr_settings.json"
        config._atomic_json(
            path,
            {
                "render_scale": 0.4,
                "min_extent": 640,
                "release": 100,
                "other_tool": {"value": "preserve me"},
            },
        )
        observed = state.get_settings(self.root)
        assert observed["settings"]["min_extent"] == 640
        external = config._read_json(path)
        external["detail_strength"] = 1.7
        external["other_tool"]["value"] = "new external value"
        config._atomic_json(path, external)
        with (
            patch.object(
                process,
                "_process_snapshot",
                side_effect=AssertionError("No game restart"),
            ),
            patch.object(
                process,
                "_running_executable",
                side_effect=AssertionError("No game restart"),
            ),
        ):
            result = state.save_settings(self.root, {"render_scale": 0.25})
        saved = config._read_json(path)
        assert saved["render_scale"] == 0.25
        assert saved["detail_strength"] == 1.7
        assert saved["min_extent"] == 640
        assert saved["release"] == 100
        assert saved["other_tool"]["value"] == "new external value"
        assert not result["game_restart_required"]
        assert not result["processing_confirmed"]
        assert trigger.read_bytes() == b"existing effect"
        assert state.profile_path(self.root).read_bytes() == profile_before

    def test_invalid_runtime_settings_leave_saved_bytes_untouched(self):
        path = self.root / "work/nr_settings.json"
        config._atomic_json(path, {"render_scale": 0.4, "unknown": "untouched"})
        before = path.read_bytes()
        invalid = (
            {"render_scale": 0.049},
            {"render_scale": 1.001},
            {"render_scale": True},
            {"intensity": float("nan")},
            {"detail_strength": float("inf")},
            {"colour_strength": -1},
            {"temporal": 1.01},
            {"hold": False},
            {"cut_limit": None},
            {"release": 256},
            {"min_extent": 127},
            {"min_extent": 4097},
            {"profile": "unknown"},
            {"master": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes), pytest.raises(ValueError):
                state.save_settings(self.root, changes)
            assert path.read_bytes() == before
        assert not list(path.parent.glob(".nr-json-*"))

    def test_saved_invalid_settings_are_reported_instead_of_claiming_default_values(
        self,
    ):
        path = self.root / "work/nr_settings.json"
        for values in ([0.4], {"render_scale": float("nan")}, {"hold": True}):
            with self.subTest(values=values):
                config._atomic_json(path, values)
                with pytest.raises(ValueError):
                    state.get_settings(self.root)

    def test_atomic_settings_failure_leaves_last_readable_document(self):
        path = self.root / "work/nr_settings.json"
        config._atomic_json(path, {"render_scale": 0.4})
        before = path.read_bytes()
        with (
            patch.object(Path, "replace", side_effect=OSError("fixture rename failure")),
            pytest.raises(OSError),
        ):
            state.save_settings(self.root, {"render_scale": 0.3})
        assert path.read_bytes() == before
        assert not list(path.parent.glob(".nr-json-*"))

    def test_reset_one_and_all_write_defaults_consumed_by_actual_live_settings_parser(
        self,
    ):
        # Execute just the real daemon's Settings class, avoiding its GPU imports.
        daemon_source = (Path(__file__).resolve().parents[1] / "layer/nr_daemon.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(daemon_source)
        actual_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Settings"
        )
        namespace = {
            "os": os,
            "math": math,
            "json": json,
            "nr_frame": SimpleNamespace(
                PROFILES={choice: {} for choice in ("standard", "natural", "cinematic", "neutral")}
            ),
        }
        exec(
            compile(ast.Module(body=[actual_class], type_ignores=[]), "nr_daemon.py", "exec"),
            namespace,
        )
        path = self.root / "work/nr_settings.json"
        result = state.save_settings(
            self.root, {"render_scale": 0.25, "intensity": 0.3, "profile": "natural"}
        )
        values = config._read_json(path)
        values["other_tool"] = 7
        config._atomic_json(path, values)
        args = SimpleNamespace(settings=str(path), **result["defaults"])
        live = namespace["Settings"](args)
        live.refresh()
        assert (live.render_scale, live.intensity, live.profile) == (
            0.25,
            0.3,
            "natural",
        )
        result = state.reset_settings(self.root, ["render_scale"])
        live.stamp = None  # Force observation even on a coarse-mtime test filesystem.
        live.refresh()
        assert (live.render_scale, live.intensity, live.profile) == (1, 0.3, "natural")
        assert config._read_json(path)["other_tool"] == 7
        result = state.reset_settings(self.root)
        live.stamp = None
        live.refresh()
        assert {name: getattr(live, name) for name in live.KNOBS} == result["defaults"]
        assert set(result["explicit_keys"]) == set(result["defaults"])
        assert config._read_json(path)["other_tool"] == 7

    def test_unknown_reset_does_not_change_settings_or_trigger(self):
        path = self.root / "work/nr_settings.json"
        config._atomic_json(path, {"render_scale": 0.4})
        before = path.read_bytes()
        with pytest.raises(ValueError):
            state.reset_settings(self.root, ["master", "render_scale"])
        assert path.read_bytes() == before
        assert not (self.root / "work/nr_trigger").exists()

    def test_non_regular_settings_destination_is_rejected(self):
        path = self.root / "work/nr_settings.json"
        path.mkdir()
        sentinel = path / "foreign.txt"
        sentinel.write_bytes(b"preserve")
        with pytest.raises(ValueError):
            state.save_settings(self.root, {"render_scale": 0.3})
        assert sentinel.read_bytes() == b"preserve"

    def test_staging_failure_preserves_existing_owned_installation(self):
        destination = self.own()
        sentinel = destination / "old.dll"
        sentinel.write_bytes(b"previous installation")
        with patch.object(deps, "_copy_runtime", side_effect=OSError("fixture copy failed")):
            result = manage.install(self.profile)
        assert not result["ok"]
        assert sentinel.read_bytes() == b"previous installation"
        assert not list(self.game.parent.glob(".dlss-nr-stage-*"))

    def test_late_commit_failure_rolls_back_installation_settings_profile_and_trigger(
        self,
    ):
        destination = self.own()
        sentinel = destination / "old.dll"
        sentinel.write_bytes(b"previous installation")
        state.save_profile(self.profile)
        config._atomic_json(self.root / "work/nr_settings.json", {"render_scale": 0.9})
        (self.root / "work/nr_trigger").write_bytes(b"old trigger")
        paths = [
            state.profile_path(self.root),
            self.root / "work/nr_settings.json",
            self.root / "work/nr_trigger",
        ]
        before = {path: path.read_bytes() for path in paths}
        with patch.object(state, "save_profile", side_effect=ValueError("fixture profile failure")):
            result = manage.install(self.profile)
        assert not result["ok"]
        assert sentinel.read_bytes() == b"previous installation"
        assert {path: path.read_bytes() for path in paths} == before
        assert not list(self.game.parent.glob(".dlss-nr-backup-*"))

    def test_extraction_failure_never_writes_game_even_with_success_exit_code(self):
        self.weights.unlink()
        dll = write_pe(self.base / "owned DLL (fixture)" / "nvngx_dlssnr.dll")
        profile = replace(self.profile, dll=dll)
        assert (
            "version unknown; tested with 310.8.0.0"
            in self.checks(profile)["local_model_input"]["detail"]
        )
        with patch.object(
            process,
            "_run",
            return_value={
                "returncode": 0,
                "timed_out": False,
                "output": "Done without usable weights",
            },
        ) as run:
            result = manage.install(profile)
        assert not result["ok"]
        assert "version unknown (tested: 310.8.0.0)" in result["error"]
        assert run.call_args.args[0][3] == dll
        assert not state.installed_path(profile).exists()
        # What the extractor said outlives the window's next action, for Save report.
        assert (self.root / config.WEIGHTS_LOG).read_text(
            encoding="utf-8"
        ) == "Done without usable weights"

    def test_missing_packages_only_uses_prepared_local_interpreter_consistently(self):
        local = write_pe(self.root / "work/windows-python/Scripts/python.exe")
        chosen = replace(self.profile, python=local)
        self.native.side_effect = lambda python, **kwargs: (
            python_info()
            if Path(python) == local
            else python_info(
                packages={},
                package_errors={"numpy": "missing", "safetensors": "missing"},
            )
        )
        with patch.object(
            deps,
            "ensure_dependencies",
            return_value={
                "ok": True,
                "profile": chosen.to_dict(),
                "python": str(local),
            },
        ) as prepare:
            result = manage.install(self.profile)
        assert result["ok"], result
        prepare.assert_called_once()
        assert state.load_profile(self.root).python == local
        assert state.runtime_env(config.Profile.from_dict(result["profile"]))["NR_PYTHON"] == str(
            local
        )

    def test_prepare_python_venv_and_pip_are_local_argv_without_shell(self):
        calls = []

        def run(command, **kwargs):
            calls.append([str(arg) for arg in command])
            if "venv" in command:
                directory = Path(command[-1])
                write_pe(directory / "Scripts/python.exe")
                (directory / "pyvenv.cfg").write_text("home = fixture\n")
            return {"returncode": 0, "timed_out": False, "output": "fixture"}

        with patch.object(process, "_run", side_effect=run):
            result = deps.ensure_dependencies(self.profile)
        assert result["ok"], result
        local = self.root / "work/windows-python/Scripts/python.exe"
        assert calls[0][:4] == [str(self.python), "-I", "-m", "venv"]
        assert calls[1][:4] == [str(local), "-I", "-m", "pip"]
        assert "--only-binary=:all:" in calls[1]
        assert result["profile"]["python"] == str(local)

    def test_foreign_local_python_directory_is_preserved(self):
        directory = self.root / "work/windows-python"
        directory.mkdir()
        sentinel = directory / "foreign.txt"
        sentinel.write_bytes(b"preserve")
        with patch.object(process, "_run") as run:
            assert not deps.ensure_dependencies(self.profile)["ok"]
        run.assert_not_called()
        assert sentinel.read_bytes() == b"preserve"

    def test_install_nr_returns_steams_launch_options_first(self):
        # An earlier setup could leave Steam starting a game through NR's wrapper. That no
        # longer holds the profile; Install NR returns those options before it installs.
        import windows_launch as launch
        import windows_wizard as bridge

        state.save_profile(self.profile)
        config._atomic_json(
            self.root / "work/windows-wizard/steam-backup.json", {"restored": False}
        )
        other = replace(self.profile, game_exe=write_pe(self.base / "other game" / "other.exe"))
        state.save_profile(other)
        assert state.load_profile(self.root) == other
        with (
            patch.object(
                launch,
                "restore_steam",
                return_value={"ok": False, "error": "A Steam game is running"},
            ),
            patch.object(manage, "install") as install,
        ):
            assert bridge.install_nr(other, None)["error"] == "A Steam game is running"
        install.assert_not_called()
        with patch.object(
            launch, "restore_steam", return_value={"ok": True, "changed": True}
        ) as restore:
            result = bridge.install_nr(other, None)
        assert result["ok"], result
        assert result["steam_restored"]
        assert restore.call_args.args[0] == other
        assert state.installed_path(other).exists()

    def test_install_blocks_active_selected_game_and_other_root_session(self):
        self.launch_state()
        assert not manage.install(self.profile)["ok"]
        other = replace(self.profile, game_exe=write_pe(self.base / "other game" / "other.exe"))
        assert not manage.install(other)["ok"]
        assert not state.installed_path(other).exists()

    def test_install_rechecks_game_after_extraction_and_before_commit(self):
        self.processes.side_effect = [
            [],
            [],
            [{"ProcessId": 12, "ExecutablePath": str(self.game)}],
        ]
        assert not manage.install(self.profile)["ok"]
        assert not state.installed_path(self.profile).exists()
        assert not list(self.game.parent.glob(".dlss-nr-stage-*"))

    def test_owned_trigger_off_on_does_not_claim_processing(self):
        self.own()
        assert manage.set_effect(self.profile, False)["ok"]
        enabled = manage.set_effect(self.profile, True)
        assert enabled["ok"]
        assert not enabled["processing_confirmed"]
        assert (self.root / "work/nr_trigger").is_file()
        assert manage.set_effect(self.profile, False)["ok"]
        assert not (self.root / "work/nr_trigger").exists()

    def test_directory_trigger_returns_failure_without_exception(self):
        self.own()
        (self.root / "work/nr_trigger").mkdir()
        result = manage.set_effect(self.profile, True)
        assert not result["ok"]
        assert (self.root / "work/nr_trigger").is_dir()

    def test_broken_link_trigger_cannot_write_outside_release(self):
        self.own()
        outside = self.base / "outside-trigger"
        trigger = self.root / "work/nr_trigger"
        try:
            trigger.symlink_to(outside)
        except OSError as error:
            self.skipTest("Creating symlinks is unavailable: " + str(error))
        assert not manage.set_effect(self.profile, True)["ok"]
        assert not outside.exists()

    def test_other_profile_cannot_toggle_active_root_game(self):
        self.launch_state()
        other = replace(self.profile, game_exe=write_pe(self.base / "other game" / "other.exe"))
        self.own(other)
        trigger = self.root / "work/nr_trigger"
        trigger.write_bytes(b"active A")
        assert not manage.set_effect(other, False)["ok"]
        assert trigger.read_bytes() == b"active A"

    def test_status_persists_between_processes_and_excludes_previous_launch(self):
        log = self.root / "work/nr_daemon.log"
        historical = b"half rounding: nearest\nmodel ready in 1s\nkeeping NumPy's large blocks\n800x450 rgba in 1.0s change 0.1 network 320x320\n"
        log.write_bytes(historical)
        self.launch_state(offset=len(historical))
        first = status.status(self.profile)
        assert first["processed"] == 0
        assert first["half_probe"] is None
        assert not first["allocator"]
        assert not first["fresh_frames"]
        status._LOG_STATES.clear()  # each GUI poll starts a separate Python process
        with log.open("ab") as handle:
            handle.write(
                b"half rounding: nearest\nmodel ready in 1s\nkeeping NumPy's large blocks\n800x450 rgba in 0.2s change 0.1 network 320x320\nframe rejected/failed bad format\n"
            )
        second = status.status(self.profile)
        assert second["processed"] == 1
        assert second["fresh_processed"] == 1
        assert second["fresh_frames"]
        assert second["network_shape"] == [320, 320]
        assert second["half_probe"]["ok"]
        assert second["allocator"]
        assert second["rejected"] == 1
        status._LOG_STATES.clear()
        third = status.status(self.profile)
        assert third["processed"] == 1
        assert third["fresh_processed"] == 0
        assert not third["fresh_frames"]
        self.launch_state(pid=124, offset=log.stat().st_size, started="new session")
        status._LOG_STATES.clear()
        fourth = status.status(self.profile)
        assert fourth["processed"] == 0
        assert fourth["half_probe"] is None
        assert not fourth["allocator"]

    def test_historical_daemon_log_does_not_confirm_game_connection(self):
        (self.root / "work/nr_daemon.log").write_text(
            "800x450 rgba in 0.2s change 0.1 network 320x320\n", encoding="utf-8"
        )
        result = status.status(self.profile)
        assert result["processed"] == 1
        assert result["game_connected"] is None
        assert not result["fresh_frames"]
        assert not result["active_game"]

    def test_status_partial_lines_survive_process_restart(self):
        log = self.root / "work/nr_daemon.log"
        log.write_bytes(b"800x450 rgba in 0.2s change")
        self.launch_state()
        assert status.status(self.profile)["processed"] == 0
        status._LOG_STATES.clear()
        with log.open("ab") as handle:
            handle.write(b" 0.1 network 320x320\n")
        result = status.status(self.profile)
        assert result["processed"] == 1
        assert result["fresh_frames"]

    def test_status_read_is_bounded_and_labels_counts_incomplete(self):
        log = self.root / "work/nr_daemon.log"
        log.write_bytes(
            b"uninteresting\n" * (config.MAX_LOG_READ // 10)
            + b"800x450 rgba in 0.2s change 0.1 network 320x320\n"
        )
        result = status.status(self.profile)
        assert not result["counts_complete"]
        assert result["processed"] == 1

    def test_status_preserves_failed_probe_and_ignores_malformed_timing(self):
        (self.root / "work/nr_daemon.log").write_text(
            "half rounding: packHalf2x16 3/1024\nFAIL: rounding mismatch\nmodel ready in 1s\n800x450 rgba in 1..2s change 0.1\n"
        )
        result = status.status(self.profile)
        assert not result["half_probe"]["ok"]
        assert result["processed"] == 0
        assert result["errors"] == 1

    def test_export_without_git_or_settings_includes_diagnostics_only(self):
        config._atomic_json(
            self.root / "release-metadata.json",
            {
                "schema_version": 1,
                "source_commit": "04459d2",
                "source_branch": "improve-release",
            },
        )
        (self.root / "work/nr_daemon.log").write_text("fixture daemon output\n")
        report = self.base / "reports (local)" / "result.json"
        with (
            patch.object(
                status,
                "_system_metadata",
                return_value={"gpu": [{"Name": "fixture GPU", "DriverVersion": "fixture"}]},
            ),
            patch.object(
                process,
                "_run",
                side_effect=AssertionError("No Git or executable needed"),
            ),
        ):
            result = status.export_report(self.profile, report)
        assert result["ok"], result
        saved = config._read_json(report)
        assert saved["settings"] is None
        assert saved["settings_error"]
        assert saved["release"]["source_commit"] == "04459d2"
        assert "fixture daemon output" in saved["daemon_log_tail"]
        # What Check says, so a report of a failed installation names the item (issue #12).
        assert "runtime_dependencies" in {item["name"] for item in saved["checks"]}
        assert saved["dll"] == {"present": False}
        assert saved["weights_log_tail"] == ""
        assert "fixture_648" not in report.read_text(encoding="utf-8")
        assert not status.export_report(self.profile, self.base / "report.dll")["ok"]

    def test_a_python_or_a_library_that_fails_says_why_in_its_check(self):
        self.native.return_value = python_info(probe_ok=False, probe_error="no answer in 30 s")
        assert "did not answer: no answer in 30 s" in self.checks()["python_native_x64"]["detail"]
        self.native.return_value = python_info()
        self.deps.return_value = {
            "ok": False,
            "failures": {
                str(self.root / "work/libnr_image.dll"): "Could not find module",
                "VCOMP140.DLL": "missing: install the Microsoft Visual C++ x64 Redistributable",
            },
        }
        detail = self.checks()["runtime_dependencies"]["detail"]
        assert "libnr_image.dll: Could not find module" in detail
        assert (
            "VCOMP140.DLL: missing: install the Microsoft Visual C++ x64 Redistributable" in detail
        )


class ProbeTests(unittest.TestCase):
    """What the Python and library probes say when they fail; the probe's process stood in for."""

    def answering(self, **got):
        result = {
            "command": [],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "output": "",
            "timed_out": False,
        }
        result.update(got)
        return patch.object(process, "_run", return_value=result)

    def test_a_missing_openmp_runtime_is_named_with_its_installer(self):
        profile = SimpleNamespace(python=Path("python.exe"), root=Path("release"))
        answer = json.dumps(
            {
                "loaded": ["vulkan-1.dll"],
                "failures": {
                    "release/work/libnr_image.dll": "Could not find module",
                    "VCOMP140.DLL": "missing",
                },
            }
        )
        with self.answering(stdout=answer):
            result = deps._load_dependencies(profile)
        assert not result["ok"]
        assert "Visual C++ x64 Redistributable" in result["failures"]["VCOMP140.DLL"]
        assert config.VC_REDIST in result["failures"]["VCOMP140.DLL"]

    @unittest.skipUnless(os.name == "nt", "Windows' version resource")
    def test_a_files_version_is_read_from_its_resource(self):
        kernel = Path(os.environ["SYSTEMROOT"]) / "System32" / "kernel32.dll"
        assert re.search(r"^\d+\.\d+\.\d+\.\d+$", deps._file_version(kernel) or "")
        with tempfile.TemporaryDirectory() as folder:
            assert deps._file_version(write_pe(Path(folder) / "no version.dll")) is None
            assert deps._file_version(Path(folder) / "absent.dll") is None

    def test_probes_that_do_not_answer_say_so(self):
        profile = SimpleNamespace(python=Path("python.exe"), root=Path("release"))
        with self.answering(returncode=None, timed_out=True):
            assert deps._load_dependencies(profile)["failures"] == {
                "dependency_probe": "no answer in 30 s"
            }
            info = process._python_info(Path("python.exe"))
        assert not info["probe_ok"]
        assert info["probe_error"] == "no answer in 30 s"


class CommandTests(unittest.TestCase):
    def test_capture_both_streams_exact_argv_and_exit_code(self):
        result = process._run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys; print(sys.argv[1]); print('stderr fixture',file=sys.stderr); sys.exit(7)",
                "spaces (and parentheses)",
            ],
            timeout=5,
        )
        assert result["returncode"] == 7
        assert "spaces (and parentheses)" in result["stdout"]
        assert "stderr fixture" in result["stderr"]

    def test_deadline_applies_when_exited_parent_descendant_keeps_output_open(self):
        # Child is ours, CPU-only and self-terminates; no game or daemon is touched.
        child = "import sys,time; end=time.monotonic()+3;\nwhile time.monotonic()<end: print('x'*100,flush=True); time.sleep(.001)"
        parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-I','-c',sys.argv[1]])"
        started = time.monotonic()
        result = process._run([sys.executable, "-I", "-c", parent, child], timeout=0.1)
        assert result["timed_out"]
        assert time.monotonic() - started < 2.8
        time.sleep(1)  # the finite child finishes before this test process exits


@unittest.skipUnless(os.name == "nt", "asks Windows about a process")
class RunningExecutableTests(unittest.TestCase):
    """The status check's one query: a live process by its path, a finished one not at all."""

    def test_live_process_reports_its_executable(self):
        # Not sys.executable: a virtual environment's python.exe starts the base interpreter.
        import ctypes

        own = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetModuleFileNameW(None, own, len(own))
        assert config._canonical(process._running_executable(os.getpid())) == config._canonical(
            own.value
        )

    def test_finished_and_invalid_processes_report_nothing(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(timeout=60)
        assert process._running_executable(child.pid) is None
        for pid in (0, -1, None, True, "12"):
            assert process._running_executable(pid) is None


if __name__ == "__main__":
    unittest.main()
