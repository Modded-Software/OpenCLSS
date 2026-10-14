#!/usr/bin/env python3
"""Exercise release assembly and first-time setup without a compiler or GPU."""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
import build_release as release
import fetch_dxvk as fetch
import pytest


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.room = tempfile.TemporaryDirectory(prefix="nr-release-test-")
        self.addCleanup(self.room.cleanup)
        self.base = Path(self.room.name)
        self.root = self.base / "checkout (source)"
        self.target = self.base / "package (release)"
        # Minimal input tree with the real runtime's shader inventory and scripts.
        paths = [
            "src/gpu/libxmx.c",
            "src/gpu/xmx.py",
            "src/gpu/xmxrt/native.py",
            "src/gpu/xmxrt/__init__.py",
            "src/gpu/xmxrt/ops.py",
            "src/gpu/xmxrt/memory.py",
            "src/gpu/xmxrt/scratch.py",
            "src/gpu/xmxrt/profile.py",
            "src/gpu/xmxrt/config.py",
            "src/gpu/xmxrt/recorder.py",
            "src/gpu/xmxrt/gemm.py",
            "src/gpu/xmxrt/attention.py",
            "src/gpu/xmxrt/elementwise.py",
            "src/layer/VkLayer_dlss_nr.json",
            "src/layer/nr_knobs.py",
            "scripts/get_weights.py",
            "scripts/build_release.py",
            "tools/deploy.sh",
            "dist-tools/setup.sh",
            "dist-tools/setup.bat",
            "docs/RELEASE-QUICKSTART.md",
            "LICENSE",
            "NOTICE",
        ]
        paths += ["dist-tools/" + name for name in release.WINDOWS_UI]
        paths += ["src/tools/" + name for name in release.WINDOWS_TOOLS]
        paths += ["src/tools/wizard/" + name for name in release.WIZARD_FILES]
        for name in paths:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        for name in (
            "src/layer/nr_daemon.py",
            "src/ref/nr_frame.py",
            "src/bench/probe.py",
            "work/mlx-dlss/LICENSE",
            "work/mlx-dlss/python/mlxdlss/features.py",
            "work/mlx-dlss/python/mlxdlss/tools/extract_dlssnr_weights.py",
            "work/mlx-dlss/python/mlxdlss/tools/unpack_dlssnr_weights.py",
        ):
            self.write(name, "fixture\n")
        for name in (
            "nr_layer.dll",
            "libxmx.dll",
            "libnr_image.dll",
            "libnr_alloc.dll",
            "libnr_layer.so",
            "libxmx.so",
            "libnr_image.so",
        ):
            self.write("work/" + name, name)
        self.write("work/NR-Setup.exe", "own x64 GUI host fixture")
        self.write("work/nr_layer32.dll", "nr_layer32.dll")
        for name in release.PROXIES:
            self.write("work/" + name, name)
        for name in release.DXVK:
            self.write(name, "dxvk:" + name)
        self.write("work/dxvk/LICENSE", "zlib/libpng licence fixture")
        self.write("work/dxvk/VERSION", "3.1.1\n")
        for name in release.required_shaders(self.root):
            self.write("work/" + name, "shader:" + name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def cli(self, script, *args, environment=None):
        command = ([sys.executable] if script.endswith(".py") else ["bash"]) + [
            str(self.root / script)
        ]
        result = subprocess.run(
            command + list(map(str, args)),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.root,
            env={**os.environ, **(environment or {})},
        )
        if result.returncode != 0:
            sys.stderr.write(
                f"release command exited {result.returncode}: {result.stderr.strip()}\n"
            )
        return result

    def test_windows_payload_keeps_allocator_and_excludes_local_inputs(self):
        for name in (
            "nvngx_dlssnr.dll",
            "work/mlxw/dlssnr-logical.safetensors",
            "work/mlx-dlss/.git/config",
            "src/layer/local.dll",
            "src/ref/local.safetensors",
            "src/ref/local.pt",
            "src/bench/frame.png",
            "work/mlx-dlss/python/mlxdlss/cache.npy",
        ):
            self.write(name, "must stay local")
        release.assemble(self.root, self.target, "windows")
        assert (self.target / "work/libnr_alloc.dll").read_text() == "libnr_alloc.dll"
        assert (self.target / "nr_layer.dll").is_file()
        assert (self.target / "nr_layer32.dll").read_text() == "nr_layer32.dll"
        for name in release.PROXIES:
            assert (self.target / name).read_text() == name
        assert (self.target / "dxvk/x32/d3d9.dll").read_text() == "dxvk:work/dxvk/x32/d3d9.dll"
        assert (self.target / "dxvk/LICENSE").is_file()
        assert json.loads((self.target / "release-metadata.json").read_text())["dxvk"] == "3.1.1"
        assert (self.target / "work/half_probe.spv").is_file()
        assert (self.target / "work/mlx-dlss/LICENSE").is_file()
        assert (
            self.target / "work/mlx-dlss/python/mlxdlss/tools/unpack_dlssnr_weights.py"
        ).is_file()
        assert (
            json.loads((self.target / "work/nr_settings.json").read_text())["render_scale"] == 0.4
        )
        assert not (self.target / "work/libxmx.so").exists()
        for name in (
            "nvngx_dlssnr.dll",
            "work/mlxw",
            "work/mlx-dlss/.git",
            "src/layer/local.dll",
            "src/ref/local.safetensors",
            "src/bench/frame.png",
            "src/ref/local.pt",
            "work/mlx-dlss/python/mlxdlss/cache.npy",
        ):
            assert not (self.target / name).exists(), name

    def test_linux_payload_has_the_setup_layer_name(self):
        self.target.mkdir()
        release.assemble(self.root, self.target, "linux")
        assert (self.target / "nr_layer.so").read_text() == "libnr_layer.so"
        assert (self.target / "work/libnr_image.so").is_file()
        assert (self.target / "README.md").is_file()
        if os.name != "nt":
            assert os.access(self.target / "setup.sh", os.X_OK)
        assert not (self.target / "work/libnr_alloc.dll").exists()

    def test_windows_wizard_ships_without_git_or_local_profiles(self):
        self.write("work/windows-profile.json", "local paths must stay private")
        self.write("work/windows-wizard/steam-backup.json", "private Steam configuration")
        release.assemble(self.root, self.target, "windows")
        assert (self.target / "NR-Setup.cmd").is_file()
        assert (self.target / "NR-Setup.exe").is_file()
        assert (self.target / "windows-wizard.ps1").is_file()
        for name in release.WINDOWS_TOOLS:
            assert (self.target / "scripts" / name).is_file()
        for name in release.WIZARD_FILES:
            assert (self.target / "scripts" / "wizard" / name).is_file()
        metadata = json.loads((self.target / "release-metadata.json").read_text())
        assert metadata["platform"] == "windows"
        assert metadata["source_commit"] is None
        assert not (self.target / "work/windows-profile.json").exists()
        assert not (self.target / "work/windows-wizard").exists()

    def test_windows_release_needs_dxvk_and_the_32bit_layer(self):
        for name in (
            "work/dxvk/x64/dxgi.dll",
            "work/nr_layer32.dll",
            "work/dxvk/LICENSE",
            "work/nr_vulkan_proxy.dll",
            "work/nr_vulkan_proxy32.dll",
        ):
            path = self.root / name
            saved = path.read_bytes()
            path.unlink()
            with pytest.raises(ValueError) as caught:
                release.assemble(self.root, self.target, "windows")
            assert str(Path(name)) in str(caught.value)
            assert not self.target.exists()
            path.write_bytes(saved)
        release.assemble(self.root, self.target, "linux")
        assert not (self.target / "dxvk").exists()
        assert json.loads((self.target / "release-metadata.json").read_text())["dxvk"] is None

    def test_fetch_dxvk_takes_only_the_pinned_archive(self):
        archive = self.base / "dxvk-9.9.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for directory in fetch.DIRECTORIES:
                for name in fetch.FILES:
                    data = f"{directory}/{name}".encode()
                    info = tarfile.TarInfo(f"dxvk-9.9/{directory}/{name}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        pin = self.base / "dxvk.json"
        pin.write_text(
            json.dumps(
                {
                    "version": "9.9",
                    "url": "https://example.invalid/dxvk-9.9.tar.gz",
                    "sha256": fetch.sha256(archive),
                }
            )
        )
        target = self.base / "work/dxvk"
        fetch.fetch(target, archive, pin)
        assert (target / "x32/d3d9.dll").read_bytes() == b"x32/d3d9.dll"
        assert (target / "VERSION").read_text().strip() == "9.9"
        assert (target / "LICENSE").is_file()
        fetch.fetch(target, self.base / "absent.tar.gz", pin)  # current: the archive is not read
        pin.write_text(json.dumps({"version": "9.8", "url": "unused", "sha256": "0" * 64}))
        with pytest.raises(ValueError):
            fetch.fetch(target, archive, pin)
        assert (target / "VERSION").read_text().strip() == "9.9"

    def test_missing_runtime_fails_before_creating_a_release(self):
        for name in (
            "work/libxmx.dll",
            "work/libnr_image.dll",
            "work/libnr_alloc.dll",
            "work/half_probe.spv",
            "work/gemm_staged.spv",
            "work/mlx-dlss/python/mlxdlss/features.py",
            "work/mlx-dlss/python/mlxdlss/tools/extract_dlssnr_weights.py",
            "work/mlx-dlss/python/mlxdlss/tools/unpack_dlssnr_weights.py",
        ):
            with self.subTest(missing=name):
                path = self.root / name
                saved = path.read_bytes()
                path.unlink()
                try:
                    with pytest.raises(ValueError, match="Incomplete build") as error:
                        release.assemble(self.root, self.target, "windows")
                    assert str(Path(name)) in str(error.value)
                    assert not self.target.exists()
                finally:
                    path.write_bytes(saved)

    def test_nonempty_destination_is_preserved(self):
        self.target.mkdir()
        sentinel = self.target / "nvngx_dlssnr.dll"
        sentinel.write_bytes(b"user-owned file")
        with pytest.raises(ValueError, match="new or empty"):
            release.assemble(self.root, self.target, "windows")
        assert sentinel.read_bytes() == b"user-owned file"
        # By name: MSYS2's Python joins iterdir()'s entries with "\" and the target with "/",
        # and the two paths then compare unequal although they name the same file.
        assert [path.name for path in self.target.iterdir()] == [sentinel.name]

    def test_missing_shared_controls_catalogue_refuses_release(self):
        (self.root / "src/layer/nr_knobs.py").unlink()
        with pytest.raises(ValueError, match="nr_knobs"):
            release.assemble(self.root, self.target, "windows")
        assert not self.target.exists()

    def test_source_overlap_is_refused(self):
        for name in (
            "src/release",
            "work/release",
            "scripts/release",
            "dist-tools/release",
        ):
            with self.subTest(destination=name):
                with pytest.raises(ValueError, match="overlaps"):
                    release.assemble(self.root, self.root / name, "linux")

    def test_copy_failure_does_not_leave_a_release(self):
        with (
            mock.patch.object(release.shutil, "copy2", side_effect=OSError("copy failed")),
            pytest.raises(OSError, match="copy failed"),
        ):
            release.assemble(self.root, self.target, "linux")
        assert not self.target.exists()
        assert not list(self.base.glob(".nr-release-*"))

    def test_windows_cli_fails_on_missing_allocator_without_success_message(self):
        (self.root / "work/libnr_alloc.dll").unlink()
        got = self.cli("scripts/build_release.py", self.target, "--platform", "windows")
        assert got.returncode == 3, got.stdout + got.stderr
        assert "libnr_alloc.dll" in got.stderr
        assert "Done." not in got.stdout
        assert not self.target.exists()

    @unittest.skipIf(os.name == "nt", "Linux deploy wrapper needs bash")
    def test_linux_deploy_release_works_without_game_or_weights(self):
        got = self.cli(
            "tools/deploy.sh",
            "--release",
            self.target,
            "--skip-build",
            "--skip-weights",
        )
        assert got.returncode == 0, got.stdout + got.stderr
        assert (self.target / "nr_layer.so").is_file()
        assert not (self.target / "work/mlxw").exists()

    @unittest.skipIf(os.name == "nt", "Linux setup needs bash")
    def test_linux_setup_checks_weights_before_installing(self):
        release.assemble(self.root, self.target, "linux")
        game = self.base / "game (empty)"
        game.mkdir()
        got = subprocess.run(
            [str(self.target / "setup.sh"), "--game", str(game), "--skip-weights"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.target,
        )
        assert got.returncode == 3, got.stdout + got.stderr
        assert not (game / "dlss-nr").exists()

    @unittest.skipIf(os.name == "nt", "Linux setup needs bash")
    def test_linux_setup_dry_run_preserves_the_game(self):
        release.assemble(self.root, self.target, "linux")
        weights = self.target / "work/mlxw/dlssnr-logical.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"local extraction fixture")
        game = self.base / "game (dry run)"
        game.mkdir()
        got = subprocess.run(
            [
                str(self.target / "setup.sh"),
                "--game",
                str(game),
                "--skip-weights",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.target,
        )
        assert got.returncode == 0, got.stdout + got.stderr
        assert list(game.iterdir()) == []

    @unittest.skipIf(os.name == "nt", "Linux setup needs bash")
    def test_linux_setup_honours_the_selected_python_for_extraction(self):
        release.assemble(self.root, self.target, "linux")
        game = self.base / "game (extraction)"
        game.mkdir()
        dll = self.base / "own files (input)" / "nvngx_dlssnr.dll"
        dll.parent.mkdir()
        dll.write_bytes(b"fixture, not a vendor DLL")
        interpreter = self.base / "chosen python (fixture)"
        interpreter.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
            'if sys.argv[1] != "-c":\n'
            '    Path(os.environ["NR_TEST_PYTHON_REPORT"]).write_text(json.dumps(sys.argv))\n'
            '    work = Path(sys.argv[sys.argv.index("--work-dir") + 1])\n'
            '    (work / "mlxw").mkdir()\n'
            '    (work / "mlxw/dlssnr-logical.safetensors").write_bytes(b"fixture")\n'
        )
        interpreter.chmod(0o755)
        report = self.base / "interpreter.json"
        env = {
            **os.environ,
            "NR_PYTHON": str(interpreter),
            "NR_TEST_PYTHON_REPORT": str(report),
        }
        got = subprocess.run(
            [str(self.target / "setup.sh"), "--game", str(game), "--dll", str(dll), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.target,
            env=env,
        )
        assert got.returncode == 0, got.stdout + got.stderr
        arguments = json.loads(report.read_text())
        assert arguments[0] == str(interpreter)
        assert arguments[2] == str(dll)
        assert arguments[-1] == str(self.target / "work")
        assert list(game.iterdir()) == []

    @unittest.skipIf(os.name == "nt", "Linux setup needs bash")
    def test_linux_setup_installs_and_launches_with_the_release_paths(self):
        release.assemble(self.root, self.target, "linux")
        weights = self.target / "work/mlxw/dlssnr-logical.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"local extraction fixture")
        game = self.base / "game (new) ' $"
        game.mkdir()
        got = subprocess.run(
            [
                str(self.target / "setup.sh"),
                "--game",
                str(game),
                "--name",
                "test_layer.so",
                "--skip-weights",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.target,
        )
        assert got.returncode == 0, got.stdout + got.stderr
        installed = game / "dlss-nr"
        manifest = json.loads((installed / "VkLayer_dlss_nr.json").read_text())
        assert manifest["layer"]["library_path"] == "./test_layer.so"
        assert (installed / "test_layer.so").read_bytes() == (
            self.target / "nr_layer.so"
        ).read_bytes()
        assert (installed / "work/half_probe.spv").is_file()
        assert not (installed / "work/mlxw").exists()
        executable = game / "game.py"
        executable.write_text(
            "#!/usr/bin/env python3\nimport json, os\nfrom pathlib import Path\n"
            'keys = ("NR_ROOT", "NR_SETTINGS", "NR_LAYER_LOG", "NR_LAYER_TRIGGER")\n'
            'Path(os.environ["NR_TEST_LAUNCH_REPORT"]).write_text('
            "json.dumps({key: os.environ[key] for key in keys}))\n"
        )
        executable.chmod(0o755)
        report = self.base / "launch.json"
        got = subprocess.run(
            [str(installed / "launch-nr.sh")],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **os.environ,
                "GAME_EXE": str(executable),
                "NR_TEST_LAUNCH_REPORT": str(report),
            },
        )
        assert got.returncode == 0, got.stdout + got.stderr
        paths = json.loads(report.read_text())
        assert paths["NR_ROOT"] == str(self.target)
        assert paths["NR_SETTINGS"] == str(self.target / "work/nr_settings.json")
        assert paths["NR_LAYER_LOG"] == str(self.target / "work/nr_daemon.log")
        assert paths["NR_LAYER_TRIGGER"] == str(self.target / "work/nr_trigger")


if __name__ == "__main__":
    unittest.main(verbosity=2)
