"""Subprocess capture, the Python and PE probes, and Windows process inspection."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from wizard import config, state


def _hidden_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _run(command, timeout=10, env=None, emit=None, cwd=None):
    """Capture both bounded streams, emitting progress without shell expansion."""
    command = [str(value) for value in command]
    result = {
        "command": command,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "output": "",
        "timed_out": False,
    }
    sys.stderr.write(f"[wizard] running: {subprocess.list2cmdline(command)}\n")
    try:
        process = subprocess.Popen(
            command,
            stdin=open(os.devnull, "rb"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env or state._clean_environment(),
            cwd=cwd,
            **_hidden_options(),
        )
    except OSError as error:
        result["stderr"] = result["output"] = str(error)
        return result
    events = queue.Queue(maxsize=256)

    def read_stream(name, handle):
        try:
            while True:
                chunk = handle.read1(4096)
                if not chunk:
                    break
                events.put((name, chunk))
        finally:
            handle.close()
            events.put((name, None))

    threads = [
        threading.Thread(target=read_stream, args=(name, handle), daemon=True)
        for name, handle in (("stdout", process.stdout), ("stderr", process.stderr))
    ]
    for worker in threads:
        worker.start()
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    completed = set()
    deadline = time.monotonic() + timeout
    while len(completed) < 2:
        if result["timed_out"] and time.monotonic() > deadline + 2:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not result["timed_out"]:
            result["timed_out"] = True
            if process.poll() is None:
                process.kill()  # only the subprocess this invocation created
        try:
            name, chunk = events.get(timeout=0.1)
        except queue.Empty:
            if result["timed_out"] and time.monotonic() > deadline + 2:
                break
            continue
        if chunk is None:
            completed.add(name)
            continue
        room = config.MAX_OUTPUT - len(chunks[name])
        if room > 0:
            chunks[name].extend(chunk[:room])
        if emit:
            try:
                emit(chunk.decode("utf-8", errors="replace"))
            except Exception as error:
                sys.stderr.write(f"[wizard] progress callback failed: {error}\n")
    try:
        result["returncode"] = process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        result["returncode"] = process.wait(timeout=2)
    for name, content in chunks.items():
        result[name] = bytes(content).decode("utf-8", errors="replace")
    result["output"] = result["stdout"] + result["stderr"]
    if result["timed_out"]:
        result["output"] += f"\nCommand timed out after {timeout:g}s"
    return result


_PYTHON_PROBE = """
import ctypes, json, os, struct, sys, sysconfig
result = {'platform': sys.platform, 'os_name': os.name, 'bits': struct.calcsize('P') * 8,
          'executable': sys.executable, 'version': sys.version.split()[0],
          'abi_platform': sysconfig.get_platform(),
          'native_apis': ('WinDLL' in dir(ctypes)) and ('add_dll_directory' in dir(os)),
          'base_prefix': sys.base_prefix, 'packages': {}, 'package_errors': {}}
try:
    import numpy
    result['packages']['numpy'] = numpy.__version__ if '__version__' in dir(numpy) else 'present'
except Exception as error:
    result['package_errors']['numpy'] = str(error)
try:
    import safetensors
    result['packages']['safetensors'] = safetensors.__version__ if '__version__' in dir(safetensors) else 'present'
except Exception as error:
    result['package_errors']['safetensors'] = str(error)
print(json.dumps(result))
"""


def _python_info(python, timeout=30):
    # Long enough for numpy's first import, while Windows' scanner reads its 30 MB OpenBLAS for
    # the first time: at 4 s a fresh environment failed Python, its packages and the DLLs at
    # once (issue #12), and the same Python answered at once a minute later.
    got = _run([python, "-I", "-c", _PYTHON_PROBE], timeout=timeout)
    try:
        value = json.loads(got["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        value = {}
    value["probe_ok"] = got["returncode"] == 0 and not got["timed_out"]
    value["output"] = got["output"]
    if not value["probe_ok"]:
        value["probe_error"] = (
            f"no answer in {timeout} s"
            if got["timed_out"]
            else got["output"].strip()[-300:] or f"exit code {got['returncode']}"
        )
    return value


def _native_python(info):
    return bool(
        info.get("probe_ok")
        and info.get("platform") == "win32"
        and info.get("os_name") == "nt"
        and info.get("bits") == 64
        and info.get("abi_platform") == "win-amd64"
        and info.get("native_apis")
    )


def pe_architecture(path):
    """Read PE headers only; no executable or vendor DLL is loaded here."""
    try:
        with Path(path).open("rb") as handle:
            dos = handle.read(64)
            if len(dos) != 64 or dos[:2] != b"MZ":
                return None
            offset = struct.unpack_from("<I", dos, 60)[0]
            if not 64 <= offset <= 1024 * 1024:
                return None
            handle.seek(offset)
            header = handle.read(26)
        if len(header) != 26 or header[:4] != b"PE\0\0":
            return None
        machine = struct.unpack_from("<H", header, 4)[0]
        magic = struct.unpack_from("<H", header, 24)[0]
        if machine == 0x8664 and magic == 0x20B:
            return "x64"
        if machine == 0x14C and magic == 0x10B:
            return "x86"
        return f"machine-0x{machine:x}"
    except OSError as error:
        sys.stderr.write(f"[wizard] cannot read the PE header: {error}\n")
        return None


def discover_python():
    """Prefer native CPython over the MinGW toolchain, within a ten-second budget."""
    deadline = time.monotonic() + 9.5
    candidates = [os.environ.get("NR_PYTHON"), sys.executable]
    for name in ("python", "python3"):
        candidates.append(shutil.which(name))
    launcher = shutil.which("py")
    if launcher:
        got = _run([launcher, "-0p"], timeout=2)
        for line in got["stdout"].splitlines():
            match = re.search(r"([A-Za-z]:[\\/].*\.exe)\s*$", line)
            if match:
                candidates.append(match.group(1).strip().strip('"'))
    if os.name == "nt":
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, r"Software\Python\PythonCore") as versions:
                        for index in range(winreg.QueryInfoKey(versions)[0]):
                            name = winreg.EnumKey(versions, index)
                            try:
                                with winreg.OpenKey(versions, name + r"\InstallPath") as key:
                                    candidates.append(
                                        str(winreg.QueryValue(key, None)) + "\\python.exe"
                                    )
                            except OSError as error:
                                sys.stderr.write(
                                    f"[wizard] Python {name} has no InstallPath: {error}\n"
                                )
                                continue
                except OSError as error:
                    sys.stderr.write(f"[wizard] cannot enumerate Python versions: {error}\n")
                    continue
        except ImportError as error:
            sys.stderr.write(f"[wizard] winreg unavailable: {error}\n")
    existing = {}
    for value in candidates:
        if value and Path(value).is_file() and pe_architecture(value) == "x64":
            existing.setdefault(config._canonical(value), str(Path(value).resolve()))
    ordered = sorted(
        existing.values(),
        key=lambda value: (
            any(word in value.lower() for word in ("msys", "mingw", "cygwin")),
            config._canonical(value)
            != config._canonical(os.environ.get("NR_PYTHON") or sys.executable),
            value.lower(),
        ),
    )
    found = []
    for value in ordered:
        remaining = deadline - time.monotonic()
        if remaining < 0.2:
            break
        if _native_python(_python_info(value, timeout=min(2, remaining))):
            found.append(value)
    return found


def _shader_names(root):
    names = set(config.SHADERS)
    for name in ("src/gpu/libxmx.c", "src/gpu/xmx.py", "src/gpu/xmxrt/native.py"):
        path = root / name
        if path.is_file():
            names.update(re.findall(r"([A-Za-z0-9_]+\.spv)", path.read_text(encoding="utf-8")))
    return sorted(names)


def _process_snapshot():
    if os.name != "nt":
        raise ValueError("Windows process inspection is required before installation")
    executable = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Select-Object ProcessId,Name,ExecutablePath) | ConvertTo-Json -Compress"
    got = _run([executable, "-NoProfile", "-NonInteractive", "-Command", script], timeout=4)
    if got["returncode"] != 0 or got["timed_out"]:
        raise ValueError("Could not safely inspect running games: " + got["output"])
    parsed = json.loads(got["stdout"] or "[]")
    return parsed if isinstance(parsed, list) else [parsed]


def _running_executable(pid):
    """The executable of the process `pid` while it is still running, else None.

    One handle and two queries. The status check runs every 5 s while the window is in
    front, and the whole WMI process list it used to take cost 0.45 s of CPU each time, in
    a PowerShell started for it, while the game played."""
    if os.name != "nt":
        raise ValueError("Windows process inspection is required")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if (
            not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259
        ):  # STILL_ACTIVE
            return None
        size = wintypes.DWORD(32768)
        name = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            return None
        return name.value
    finally:
        kernel32.CloseHandle(handle)


def _launch_state(profile):
    path = profile.root / "work/windows-wizard/launch-state.json"
    if not path.exists():
        return {}
    value = config._read_json(path, limit=128 * 1024)
    if not isinstance(value, dict):
        raise ValueError("Launch state must be a JSON object")
    return value


def _install_process_guard(profile):
    try:
        processes = _process_snapshot()
        state_value = _launch_state(profile)
        game_directory = config._canonical(profile.game_exe.parent)
        for process in processes:
            executable = process.get("ExecutablePath")
            if executable and config._canonical(executable).startswith(game_directory + "/"):
                return False, "Close the selected game before changing its installation"
            if (
                not executable
                and str(process.get("Name", "")).casefold() == profile.game_exe.name.casefold()
            ):
                return (
                    False,
                    "A matching game process could not be inspected; close it before installation",
                )
            if (
                state_value.get("pid") == process.get("ProcessId")
                and state_value.get("game_exe")
                and executable
            ):
                if config._canonical(executable) == config._canonical(state_value["game_exe"]):
                    return (
                        False,
                        "Another game is still using this release; close it before installation",
                    )
        return True, "No selected or release-owned game is running"
    except (OSError, ValueError, TypeError) as error:
        return False, str(error)
