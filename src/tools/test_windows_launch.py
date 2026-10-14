"""CPU-only launch/Steam regressions. No Steam, game, or GPU is started."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import windows_launch as launch
from wizard import deps, manage, state
from wizard.config import Profile


def localconfig(options="+set test 1", app_id="379720", newline="\r\n"):
    option = "" if options is None else '\t"LaunchOptions" ' + launch._vdf_quote(options)
    return (
        (
            '// retain this comment\n"UserLocalConfigStore" { "Software" { "Valve" { "Steam" {\n'
            '"Unrelated" "a\\\\b"\n"apps" {\n"' + app_id + '" {\n'
            '"LastPlayed" "123456"\n' + option + "\n}\n"
            '"999" { "Other" "unchanged" }\n} } } } }\n'
        )
        .replace("\n", newline)
        .encode("utf-8")
    )


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nr-launch-tests-")
        self.addCleanup(self.temporary.cleanup)
        # Resolved, as the launcher resolves what it is given: a TEMP in 8.3 form
        # (C:\Users\NAME~1\...) otherwise compares unequal to the same file's long name.
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "package (候補) & test"
        self.game = self.base / "library (games)" / "steamapps/common/DOOM"
        self.game.mkdir(parents=True)
        self.exe = self.game / "DOOMx64vk.exe"
        self.default_exe = self.game / "DOOMx64.exe"
        self.python = self.base / "Python (custom) & 日本語/python.exe"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"test interpreter")
        self.exe.write_bytes(b"selected x64 game")
        self.default_exe.write_bytes(b"Steam default x64 game")
        self.dll = self.base / "my dll (own)/nvngx_dlssnr.dll"
        self.dll.parent.mkdir(parents=True)
        self.dll.write_bytes(b"own DLL")
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "windows_launch.py").write_text("# packaged wrapper", encoding="utf-8")
        self.profile = Profile(
            self.root,
            self.python,
            self.exe,
            self.dll,
            game_args=["+set", "r_width", "800", "literal & $(text)"],
        )
        self._install_marker(self.profile)
        self.trigger = self.root / "work/nr_trigger"
        self.trigger.parent.mkdir(parents=True, exist_ok=True)
        self.trigger.write_bytes(b"on")
        self.daemon = self.root / "work/nr_daemon.log"
        self.daemon.write_bytes(b"previous session\n")
        self.addCleanup(patch.stopall)
        self.validate = patch.object(deps, "validate", return_value={"ok": True}).start()
        self.runtime_env = patch.object(state, "runtime_env", side_effect=self._environment).start()
        self.set_effect = patch.object(manage, "set_effect", side_effect=self._effect).start()
        self.save_profile = patch.object(
            state,
            "save_profile",
            side_effect=AssertionError("Steam must not overwrite GUI profile"),
        ).start()
        self.processes = patch.object(launch, "_processes", return_value=[]).start()
        patch.object(launch, "_registry", return_value=None).start()
        self.process = Mock(pid=456)
        self.process.poll.return_value = None
        self.process.wait.return_value = 0
        self.popen = patch.object(launch.subprocess, "Popen", return_value=self.process).start()
        patch.object(
            launch.subprocess,
            "run",
            side_effect=AssertionError("No real commands in tests"),
        ).start()
        self.stop = patch.object(launch, "_stop_steam", return_value=False).start()
        self.restart = patch.object(launch, "_restart_steam", return_value=789).start()
        self.steam = self.base / "Steam (client)/steam.exe"
        self.steam.parent.mkdir(parents=True)
        self.steam.write_bytes(b"fake steam")
        self.config = self.steam.parent / "userdata/123/config/localconfig.vdf"
        self.config.parent.mkdir(parents=True)
        self.config.write_bytes(localconfig())
        self.manifest = self.game.parents[1] / "appmanifest_379720.acf"
        self.manifest.write_text(
            '"AppState" { "appid" "379720" "installdir" "DOOM" }', encoding="utf-8"
        )

    def _install_marker(self, profile):
        directory = profile.game_exe.parent / "dlss-nr"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".windows-wizard-owner.json").write_text(
            json.dumps({"schema_version": 1, "root": str(profile.root)}),
            encoding="utf-8",
        )

    def _environment(self, profile):
        return {
            "NR_ROOT": str(profile.root),
            "NR_PYTHON": str(profile.python),
            "NR_LAYER_SOCKET": r"\\.\pipe\nr_windows_fixture",
            "NR_LAYER_TRIGGER": str(self.trigger),
            "NR_LAYER_LOG": str(self.daemon),
            "VK_LAYER_PATH": str(profile.game_exe.parent / "dlss-nr"),
            "ENABLE_NR_LAYER": "1",
            "NORMAL": "retained",
        }

    def _effect(self, profile, enabled):
        assert not enabled
        self.trigger.unlink(missing_ok=True)
        return {"ok": True}

    def _configure(self, profile=None):
        return launch.configure_steam(profile or self.profile, self.steam)

    def test_direct_argv_env_logs_and_fresh_effect_off_without_git(self):
        def started(command, **kwargs):
            kwargs["stdout"].write(b"game stdout\n")
            kwargs["stderr"].write(b"game stderr\n")
            return self.process

        self.popen.side_effect = started
        result = launch.launch(self.profile)
        assert result["ok"], result
        command, kwargs = self.popen.call_args
        assert command[0] == [str(self.exe), *self.profile.game_args]
        assert not kwargs["shell"]
        assert kwargs["cwd"] == str(self.game)
        assert kwargs["env"]["NR_PYTHON"] == str(self.python)
        assert kwargs["env"]["NORMAL"] == "retained"
        assert result["pid"] == 456
        assert result["daemon_start_bytes"] == len(b"previous session\n")
        assert Path(result["stdout_log"]).read_bytes() == b"game stdout\n"
        assert Path(result["stderr_log"]).read_bytes() == b"game stderr\n"
        assert not self.trigger.exists()
        assert not (self.root / ".git").exists()

    def test_running_selected_game_preserves_trigger_without_launch(self):
        self.processes.return_value = [
            {"Name": self.exe.name, "ProcessId": 100, "ExecutablePath": str(self.exe)}
        ]
        result = launch.launch(self.profile)
        assert not result["ok"]
        self.popen.assert_not_called()
        self.set_effect.assert_not_called()
        assert self.trigger.read_bytes() == b"on"

    def test_running_previous_game_sharing_root_preserves_state_and_trigger(self):
        other = self.base / "another game (B)/game.exe"
        other.parent.mkdir()
        other.write_bytes(b"another x64 game")
        profile_b = replace(self.profile, game_exe=other)
        self._install_marker(profile_b)
        state_path = self.root / "work/windows-wizard/launch-state.json"
        state_path.parent.mkdir()
        state = {"root": str(self.root), "game_exe": str(self.exe), "pid": 100}
        initial = json.dumps(state).encode("utf-8")
        state_path.write_bytes(initial)
        self.processes.return_value = [
            {"Name": self.exe.name, "ProcessId": 101, "ExecutablePath": str(self.exe)}
        ]
        result = launch.launch(profile_b)
        assert not result["ok"]
        assert state_path.read_bytes() == initial
        assert self.trigger.read_bytes() == b"on"
        self.set_effect.assert_not_called()
        self.popen.assert_not_called()

    def test_uninspectable_selected_game_and_wrong_owner_are_refused(self):
        self.processes.return_value = [
            {"Name": self.exe.name, "ProcessId": 100, "ExecutablePath": None}
        ]
        assert not launch.launch(self.profile)["ok"]
        self.processes.return_value = []
        owner = self.game / "dlss-nr/.windows-wizard-owner.json"
        owner.write_text(
            json.dumps({"schema_version": 1, "root": str(self.base / "foreign")}),
            encoding="utf-8",
        )
        assert not launch.launch(self.profile)["ok"]
        self.set_effect.assert_not_called()
        self.popen.assert_not_called()

    def test_runtime_interpreter_mismatch_never_launches(self):
        environment = self._environment(self.profile)
        environment["NR_PYTHON"] = "another-python.exe"
        self.runtime_env.side_effect = None
        self.runtime_env.return_value = environment
        assert not launch.launch(self.profile)["ok"]
        self.set_effect.assert_not_called()
        self.popen.assert_not_called()

    def test_steam_default_executable_substitution_preserves_command_tail(self):
        result, _ = launch._launch_direct(
            self.profile,
            original_command=[str(self.default_exe), "+set", "old_option", "two words"],
        )
        assert result["executable_substituted"]
        assert result["command"] == [
            str(self.exe),
            "+set",
            "old_option",
            "two words",
            *self.profile.game_args,
        ]
        assert result["steam_original_exe"] == str(self.default_exe)

    def test_foreign_original_executable_is_not_executed(self):
        foreign = self.base / "custom wrapper.exe"
        foreign.write_bytes(b"wrapper")
        with pytest.raises(launch.LaunchError):
            launch._launch_direct(self.profile, original_command=[str(foreign)])
        self.popen.assert_not_called()
        self.set_effect.assert_not_called()

    def test_immediate_failure_is_reported_with_logs_and_exit_code(self):
        self.process.poll.return_value = 5
        result = launch.launch(self.profile)
        assert not result["ok"]
        assert result["exit_code"] == 5
        assert Path(result["stderr_log"]).is_file()

    def test_quoting_round_trip_for_non_ascii_shell_characters_and_backslashes(self):
        args = [
            str(self.python),
            str(self.root / "scripts/windows_launch.py"),
            "a & $(b); %HOME%",
            "D:\\end\\",
            'embedded "quote"',
            "",
            "two words",
            "日本語 (test)",
        ]
        command = " ".join(launch._quote(arg) for arg in args)
        assert launch._split_commandline(command) == args

    def test_wrapper_snapshot_keeps_exact_selected_python(self):
        path = launch._ensure_steam_profile(self.profile)
        parsed = launch._split_commandline(launch.steam_launch_options(self.profile))
        assert parsed == [
            str(self.python),
            str(self.root / "scripts/windows_launch.py"),
            "--steam-wrapper",
            str(path),
            "--",
            "%command%",
        ]
        assert json.loads(path.read_text(encoding="utf-8")) == self.profile.to_dict()
        draft = self.root / "work/windows-profile.json"
        draft.write_text('{"draft": "unrelated"}', encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["python"] == str(self.python)

    def _write_pe(self, path, machine=0x8664, magic=0x20B):
        data = bytearray(128)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 60, 64)
        data[64:68] = b"PE\0\0"
        struct.pack_into("<H", data, 68, machine)
        struct.pack_into("<H", data, 88, magic)
        path.write_bytes(data)

    def test_x64_pythonw_hides_only_wrapper_and_keeps_snapshot_and_runtime_python(self):
        snapshot = launch._ensure_steam_profile(self.profile)
        before = snapshot.read_bytes()
        profile_before = self.profile.to_dict()
        pythonw = self.python.with_name("pythonw.exe")
        self._write_pe(pythonw)
        options = launch._split_commandline(launch.steam_launch_options(self.profile))
        assert options[0] == str(pythonw)
        assert options[3] == str(snapshot)
        assert self.profile.to_dict() == profile_before
        assert snapshot.read_bytes() == before
        assert self._environment(self.profile)["NR_PYTHON"] == str(self.python)
        result, _ = launch._launch_direct(self.profile, original_command=[str(self.default_exe)])
        assert result["ok"], result
        assert self.popen.call_args.kwargs["env"]["NR_PYTHON"] == str(self.python)

    def test_missing_pythonw_uses_selected_python(self):
        assert not self.python.with_name("pythonw.exe").exists()
        assert launch._split_commandline(launch.steam_launch_options(self.profile))[0] == str(
            self.python
        )

    def test_bad_pe_or_x86_pythonw_uses_selected_python(self):
        pythonw = self.python.with_name("pythonw.exe")
        for content in (b"not PE", b"MZ"):
            with self.subTest(content=content):
                pythonw.write_bytes(content)
                assert launch._steam_python(self.profile) == self.python
        self._write_pe(pythonw, machine=0x14C, magic=0x10B)
        assert launch._steam_python(self.profile) == self.python
        self._write_pe(pythonw, machine=0x8664, magic=0x10B)
        assert launch._steam_python(self.profile) == self.python

    def test_symlink_or_directory_pythonw_uses_selected_python(self):
        pythonw = self.python.with_name("pythonw.exe")
        self._write_pe(pythonw)
        with patch.object(
            Path, "is_symlink", autospec=True, side_effect=lambda path: path == pythonw
        ):
            assert launch._steam_python(self.profile) == self.python
        pythonw.unlink()
        pythonw.mkdir()
        assert launch._steam_python(self.profile) == self.python

    def test_wrapper_error_with_no_stderr_still_records_json(self):
        snapshot = launch._ensure_steam_profile(self.profile)
        self.validate.return_value = {"ok": False, "error": "missing runtime"}
        with (
            patch.object(launch.sys, "stderr", None),
            patch("builtins.print") as printed,
        ):
            code = launch.main(["--steam-wrapper", str(snapshot), "--", str(self.default_exe)])
        assert code == 1
        printed.assert_not_called()
        failure = self.root / "work/windows-wizard/steam-wrapper-last-error.json"
        recorded = json.loads(failure.read_text(encoding="utf-8"))
        assert not recorded["ok"]
        assert recorded["error"] == "missing runtime"
        assert recorded["profile_snapshot"] == str(snapshot)
        self.popen.assert_not_called()

    def test_modified_snapshot_is_not_overwritten(self):
        path = launch._ensure_steam_profile(self.profile)
        path.write_text('{"foreign": true}', encoding="utf-8")
        result = self._configure()
        assert not result["ok"]
        assert result["launch_options"] is None
        assert path.read_text(encoding="utf-8") == '{"foreign": true}'
        self.stop.assert_not_called()

    def test_app_id_detection_in_nested_install_with_unrelated_bad_manifest(self):
        nested = self.game / "bin/x64/game.exe"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"game")
        (self.manifest.parent / "appmanifest_bad.acf").write_text('"broken" {', encoding="utf-8")
        assert launch.detect_steam_app_id(nested) == "379720"
        (self.manifest.parent / "appmanifest_duplicate.acf").write_text(
            '"AppState" { "appid" "99" "installdir" "DOOM" }', encoding="utf-8"
        )
        assert launch.detect_steam_app_id(nested) is None
        assert launch.detect_steam_app_id(self.python) is None

    def test_vdf_edit_escaping_preserves_other_bytes_bom_and_line_endings(self):
        before = b"\xef\xbb\xbf" + localconfig('+set path "D:\\game (test)\\data"')
        value = '"D:\\Python (test)\\python.exe" "D:\\候補 & test\\wrapper.py" -- %command%'
        after = launch._edit_options(before, "379720", value)
        assert launch._options(after, "379720") == value
        assert after.startswith(b"\xef\xbb\xbf")
        assert b"// retain this comment\r\n" in after
        assert b'"999" { "Other" "unchanged" }' in after
        assert launch._edit_options(after, "379720", launch._options(before, "379720")) == before

    def test_vdf_absent_option_restore_and_new_app(self):
        before = localconfig(None)
        after = launch._edit_options(before, "379720", "wrapper")
        restored = launch._edit_options(after, "379720", None)
        assert launch._options(restored, "379720") is None
        assert b'"LastPlayed" "123456"' in restored
        new_app = launch._edit_options(before, "101", "other wrapper")
        assert launch._options(new_app, "101") == "other wrapper"
        assert launch._options(new_app, "379720") is None

    def test_duplicate_vdf_key_refuses_ambiguous_options(self):
        data = localconfig().replace(b'"LastPlayed" "123456"', b'"LaunchOptions" "another"')
        with pytest.raises(launch.LaunchError):
            launch._edit_options(data, "379720", "replacement")

    def test_atomic_edit_conflicts_keep_concurrent_file(self):
        before = self.config.read_bytes()
        self.config.write_bytes(b"foreign update")
        with pytest.raises(launch.LaunchError):
            launch._atomic_write(self.config, before, b"replacement")
        assert self.config.read_bytes() == b"foreign update"
        self.config.write_bytes(before)
        with (
            patch.object(
                launch.os,
                "fsync",
                side_effect=lambda _: self.config.write_bytes(b"second update"),
            ),
            pytest.raises(launch.LaunchError),
        ):
            launch._atomic_write(self.config, before, b"replacement")
        assert self.config.read_bytes() == b"second update"
        assert list(self.config.parent.glob("*.nr-*.tmp")) == []

    def test_configure_and_restore_preserve_existing_game_args_and_fields(self):
        before = self.config.read_bytes()
        original = launch._options(before, "379720")
        result = self._configure()
        assert result["ok"], result
        assert (
            result["launch_options"] == launch.steam_launch_options(self.profile) + " " + original
        )
        assert b'"999" { "Other" "unchanged" }' in self.config.read_bytes()
        backup_path = Path(result["backup"])
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        assert backup["original_launch_options"] == original
        assert backup["profile_snapshot"] == str(launch._steam_profile_path(self.profile))
        again = self._configure()
        assert again["already_configured"]
        assert self.stop.call_count == 1
        restored = launch.restore_steam(self.profile)
        assert restored["ok"], restored
        assert self.config.read_bytes() == before
        assert json.loads(backup_path.read_text(encoding="utf-8"))["restored"]
        self.save_profile.assert_not_called()

    def test_foreign_wrappers_and_shell_options_are_preserved(self):
        for options in [
            '"custom python.exe" wrapper.py %command%',
            "cmd /c start game.exe",
            "+set test 1 & other.exe",
            '+set "unterminated',
        ]:
            with self.subTest(options=options):
                before = localconfig(options)
                self.config.write_bytes(before)
                result = self._configure()
                assert not result["ok"]
                assert result["manual"]
                assert self.config.read_bytes() == before
                assert Path(launch._split_commandline(result["launch_options"])[3]).is_file()
        self.stop.assert_not_called()

    def test_multiple_steam_accounts_return_manual_without_changes(self):
        other = self.steam.parent / "userdata/456/config/localconfig.vdf"
        other.parent.mkdir(parents=True)
        other.write_bytes(localconfig())
        before = self.config.read_bytes()
        result = self._configure()
        assert not result["ok"]
        assert result["manual"]
        assert self.config.read_bytes() == before
        self.stop.assert_not_called()

    def test_configure_game_b_preserves_game_a_wrapper_profile_and_backup(self):
        result_a = self._configure()
        assert result_a["ok"]
        before_vdf = self.config.read_bytes()
        snapshot_a = Path(result_a["profile_snapshot"])
        before_snapshot = snapshot_a.read_bytes()
        backup_path = Path(result_a["backup"])
        before_backup = backup_path.read_bytes()
        draft = self.root / "work/windows-profile.json"
        draft.write_bytes(b"GUI draft must stay unchanged")
        exe_b = self.game.parent / "Game B/game.exe"
        exe_b.parent.mkdir()
        exe_b.write_bytes(b"another game")
        (self.manifest.parent / "appmanifest_101.acf").write_text(
            '"AppState" { "appid" "101" "installdir" "Game B" }', encoding="utf-8"
        )
        result_b = self._configure(replace(self.profile, game_exe=exe_b))
        assert not result_b["ok"]
        assert "unresolved Steam backup" in result_b["error"]
        assert self.config.read_bytes() == before_vdf
        assert snapshot_a.read_bytes() == before_snapshot
        assert backup_path.read_bytes() == before_backup
        assert draft.read_bytes() == b"GUI draft must stay unchanged"
        assert self.stop.call_count == 1

    def test_changed_profile_cannot_silently_reconfigure_existing_wrapper(self):
        assert self._configure()["ok"]
        before = self.config.read_bytes()
        changed = replace(self.profile, game_args=["+set", "another", "1"])
        result = self._configure(changed)
        assert not result["ok"]
        assert self.config.read_bytes() == before

    def test_restore_conflict_preserves_foreign_launch_options_and_backup(self):
        result = self._configure()
        backup_path = Path(result["backup"])
        before_backup = backup_path.read_bytes()
        updated = launch._edit_options(self.config.read_bytes(), "379720", "user's new options")
        self.config.write_bytes(updated)
        restored = launch.restore_steam(self.profile)
        assert not restored["ok"]
        assert self.config.read_bytes() == updated
        assert backup_path.read_bytes() == before_backup
        assert self.stop.call_count == 1

    def test_restore_recovers_committed_vdf_when_metadata_update_failed(self):
        before = self.config.read_bytes()
        result = self._configure()
        backup_path = Path(result["backup"])
        self.config.write_bytes(before)
        restored = launch.restore_steam(self.profile)
        assert restored["ok"], restored
        assert restored["metadata_recovered"]
        assert not restored["changed"]
        assert json.loads(backup_path.read_text(encoding="utf-8"))["restored"]
        assert self.config.read_bytes() == before
        assert self.stop.call_count == 1

    def test_malformed_owner_and_launch_state_fail_before_trigger_or_process(self):
        owner = self.game / "dlss-nr/.windows-wizard-owner.json"
        owner.write_text("[]", encoding="utf-8")
        assert not launch.launch(self.profile)["ok"]
        self._install_marker(self.profile)
        state = self.root / "work/windows-wizard/launch-state.json"
        state.parent.mkdir()
        state.write_text("[]", encoding="utf-8")
        assert not launch.launch(self.profile)["ok"]
        assert self.trigger.read_bytes() == b"on"
        self.set_effect.assert_not_called()
        self.popen.assert_not_called()

    def test_unknown_backup_format_is_preserved(self):
        backup = self.root / "work/windows-wizard/steam-backup.json"
        backup.parent.mkdir()
        backup.write_bytes(b"[]")
        before = self.config.read_bytes()
        result = self._configure()
        assert not result["ok"]
        assert backup.read_bytes() == b"[]"
        assert self.config.read_bytes() == before
        self.stop.assert_not_called()

    def test_concurrent_launch_options_change_during_shutdown_is_preserved(self):
        changed = launch._edit_options(self.config.read_bytes(), "379720", "+set foreign 1")

        def shutdown(*args):
            self.config.write_bytes(changed)
            return True

        self.stop.side_effect = shutdown
        result = self._configure()
        assert not result["ok"]
        assert self.config.read_bytes() == changed
        self.restart.assert_called_once_with(self.steam)

    def test_steam_restart_error_does_not_hide_configured_state(self):
        self.stop.return_value = True
        self.restart.side_effect = OSError("restart failed")
        result = self._configure()
        assert not result["ok"]
        assert result["configured"]
        assert result["steam_restart_error"] == "restart failed"
        assert launch._options(self.config.read_bytes(), "379720") == result["launch_options"]

    def test_restore_restart_error_does_not_hide_restored_state(self):
        before = self.config.read_bytes()
        assert self._configure()["ok"]
        self.stop.return_value = True
        self.restart.side_effect = OSError("restart failed")
        result = launch.restore_steam(self.profile)
        assert not result["ok"]
        assert result["changed"]
        assert self.config.read_bytes() == before

    def test_steam_games_guard_uses_other_libraries_and_running_app_id(self):
        folders = self.steam.parent / "steamapps/libraryfolders.vdf"
        folders.parent.mkdir()
        library = self.base / "another library"
        folders.write_text(
            '"libraryfolders" { "1" { "path" ' + launch._vdf_quote(str(library)) + " } }",
            encoding="utf-8",
        )
        self.processes.return_value = [
            {
                "Name": "other.exe",
                "ProcessId": 100,
                "ExecutablePath": str(library / "steamapps/common/Other/other.exe"),
            }
        ]
        assert len(launch._steam_games(self.steam, self.profile)) == 1
        self.processes.return_value = []
        with patch.object(launch, "_registry", return_value=123):
            assert len(launch._steam_games(self.steam, self.profile)) == 1

    def test_active_steam_game_never_shuts_down_client(self):
        with patch.object(launch, "_steam_games", return_value=[{"ProcessId": 100}]):
            with pytest.raises(launch.LaunchError):
                ORIGINAL_STOP(self.steam, self.profile, None)
        self.popen.assert_not_called()

    def test_lingering_running_app_id_is_waited_for_and_then_refused(self):
        lingering = [{"Name": "Steam RunningAppID 379720", "ProcessId": None}]
        with (
            patch.object(launch, "_steam_games", return_value=lingering),
            patch.object(launch, "_registry", side_effect=[379720, 0]),
            patch.object(launch, "_steam_running", return_value=False),
            patch.object(launch.time, "sleep"),
        ):
            assert not ORIGINAL_STOP(self.steam, self.profile, None)
        with (
            patch.object(launch, "_steam_games", return_value=lingering),
            patch.object(launch, "_registry", return_value=379720),
            patch.object(launch.time, "sleep"),
            patch.object(launch.time, "monotonic", side_effect=[0, 1, 16]),
        ):
            with pytest.raises(launch.LaunchError):
                ORIGINAL_STOP(self.steam, self.profile, None)
        self.popen.assert_not_called()

    def test_steam_exit_is_graceful_and_never_force_kills(self):
        with (
            patch.object(launch, "_steam_games", return_value=[]),
            patch.object(launch, "_steam_running", side_effect=[True, True, False]),
            patch.object(launch.time, "sleep"),
        ):
            assert ORIGINAL_STOP(self.steam, self.profile, None)
        assert self.popen.call_args.args[0] == [str(self.steam), "-shutdown"]
        self.process.terminate.assert_not_called()
        self.process.kill.assert_not_called()

    def test_steam_shutdown_timeout_never_force_kills_or_edits(self):
        before = self.config.read_bytes()
        with (
            patch.object(launch, "_steam_games", return_value=[]),
            patch.object(launch, "_steam_running", return_value=True),
            patch.object(launch.time, "monotonic", side_effect=[0, 21]),
        ):
            with pytest.raises(launch.LaunchError):
                ORIGINAL_STOP(self.steam, self.profile, None)
        assert self.config.read_bytes() == before
        self.process.terminate.assert_not_called()
        self.process.kill.assert_not_called()

    def test_normal_steam_restart_cleans_nr_environment(self):
        with patch.dict(
            os.environ,
            {
                "NR_ROOT": "old",
                "NR_PYTHON": "old-python",
                "NORMAL": "keep",
                "ENABLE_NR_LAYER": "1",
                "VK_LAYER_PATH": "old/dlss-nr",
                "DISABLE_VK_LAYER_VALVE_steam_fossilize_1": "1",
            },
        ):
            assert ORIGINAL_RESTART(self.steam) == 456
        kwargs = self.popen.call_args.kwargs
        assert "NR_ROOT" not in kwargs["env"]
        assert "NR_PYTHON" not in kwargs["env"]
        assert "ENABLE_NR_LAYER" not in kwargs["env"]
        assert "VK_LAYER_PATH" not in kwargs["env"]
        assert kwargs["env"]["NORMAL"] == "keep"

    def test_wrapper_cli_records_exit_and_rejects_mutable_gui_profile(self):
        snapshot = launch._ensure_steam_profile(self.profile)
        self.process.wait.return_value = 7
        code = launch.main(
            [
                "--steam-wrapper",
                str(snapshot),
                "--",
                str(self.default_exe),
                "+set",
                "old",
                "1",
            ]
        )
        assert code == 7
        state = json.loads(
            (self.root / "work/windows-wizard/launch-state.json").read_text(encoding="utf-8")
        )
        assert state["exit_code"] == 7
        assert "ended_utc" in state
        draft = self.root / "work/windows-profile.json"
        draft.write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        self.popen.reset_mock()
        with patch.object(launch.sys, "stderr"):
            assert launch.main(["--steam-wrapper", str(draft), "--", str(self.default_exe)]) == 1
        self.popen.assert_not_called()
        failure = self.root / "work/windows-wizard/steam-wrapper-last-error.json"
        assert failure.is_file()
        assert (
            "immutable release snapshot" in json.loads(failure.read_text(encoding="utf-8"))["error"]
        )


ORIGINAL_STOP = launch._stop_steam
ORIGINAL_RESTART = launch._restart_steam


if __name__ == "__main__":
    unittest.main()
