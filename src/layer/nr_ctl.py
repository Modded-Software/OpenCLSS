#!/usr/bin/env python3
"""nr-ctl - drive the running daemon and layer without restarting either.

The daemon holds the model, so restarting it to try another profile costs a second and
drops the layer's connection mid-game. Everything here is either a post-network knob or
the extent, so none of it invalidates the weights and all of it can move between frames.

    src/layer/nr-ctl                    an interactive session
    src/layer/nr-ctl on                 start enhancing (touches the trigger)
    src/layer/nr-ctl off                hand the game back
    src/layer/nr-ctl set scale 0.5      run the network at half of each side
    src/layer/nr-ctl status

Start the daemon so it watches the same file:

    python3 src/layer/nr_daemon.py --settings /tmp/nr_settings.json --render-scale 0.5

The trigger is what the layer polls; the settings file is what the daemon re-reads.
Neither needs the other, so `on`/`off` work even with no settings file, and a knob can
be changed while the effect is off.

`nr-toggle` is the same three files bound to a key, for when the game has the screen and
a terminal would mean alt-tabbing out of it.
"""

import json
import os
import pathlib
import sys
import textwrap

import cv2
import nr_knobs
import nr_paths

SETTINGS, TRIGGER, SOCKET = nr_paths.SETTINGS, nr_paths.TRIGGER, nr_paths.SOCKET
read, write, alive = nr_paths.read, nr_paths.write, nr_paths.alive

# One table, shared with `nr-panel` and with the manual: a knob that exists in the daemon
# and not here is exactly the drift `nr_knobs` was written to stop, and it had already
# happened once (`hold`).
PROFILES = nr_knobs.PROFILES
KNOBS = tuple(knob.name for knob in nr_knobs.KNOBS)
ALIAS = {
    "p": "profile",
    "i": "intensity",
    "d": "detail_strength",
    "c": "colour_strength",
    "s": "render_scale",
    "scale": "render_scale",
    "detail": "detail_strength",
    "colour": "colour_strength",
    "color": "colour_strength",
    "t": "temporal",
    "history": "temporal",
    "cut": "cut_limit",
    "h": "hold",
}

# Measured end to end on this machine, 2026-09-10 - the whole round trip through the
# socket, not the graph alone. `(swapchain width, height, render_scale) -> ms`.
# The network's cost follows the extent it is given and nothing else, but at these sizes
# the numpy either side of it - feature assembly and composition, both at the output
# resolution - is the larger half, so the swapchain size matters as much as the scale.
RATES = nr_knobs.RATES  # measured; see src/bench/live_rates.py


def _opencv():
    """Whether the vendored blur will find its fast path."""
    return cv2 is not None


FIELDS_EXTENT = __import__("re").compile(r"^\d+x\d+ ")


def status():
    values = read()
    print("  daemon    {}  ({})".format("listening" if alive() else "not answering", SOCKET))
    print("  effect    {}  ({})".format("ON" if TRIGGER.exists() else "off", TRIGGER))
    print("  settings  %s" % (SETTINGS if SETTINGS.exists() else f"{SETTINGS} - not written yet"))
    print(
        "  blur      %s; only used when detail/colour strength differs from 1"
        % (
            "OpenCV available in this interpreter"
            if _opencv()
            else "NumPy path in this interpreter"
        )
    )
    for knob in KNOBS:
        shown = values.get(knob, "(daemon default)")
        note = ""
        if knob == "temporal" and isinstance(shown, (int, float)):
            note = (
                "   off - every frame is drawn alone"
                if not float(shown)
                else "   the previous frame is carried forward, gated per pixel"
            )
        if knob == "hold" and isinstance(shown, (int, float)):
            note = (
                "   off - the model's own gate alone"
                if not float(shown)
                else "   pixels the game did not move are held still"
            )
        print("    %-16s %s%s" % (knob, shown, note))


def _version(root):
    """A packaged release does not need Git installed or a checkout beside it."""
    import subprocess

    try:
        metadata = json.loads((root / "release-metadata.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        metadata = None
    if isinstance(metadata, dict):
        commit = metadata.get("source_commit")
        if isinstance(commit, str) and commit:
            branch = metadata.get("source_branch")
            return "release {}{}".format(
                commit[:12],
                f" ({branch})" if isinstance(branch, str) and branch else "",
            )
        return "packaged release (commit not recorded)"
    try:
        version = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%h %cs", "-1"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"not a git clone (Git unavailable or timed out): {error}"
    if version.returncode != 0:
        reason = version.stderr.strip()
        return "not a git clone (%s)" % (reason or "git log failed")
    version_text = version.stdout.strip()
    if not version_text:
        return "not a git clone"
    return version_text


def report(lines=8):
    """Everything a bug report needs, in one paste.

    Somebody whose frames quietly stop being processed sees a game that looks normal and a
    log they have no reason to know about. This is the command to ask them for: what is
    running, on which GPU, with which knobs, and what the daemon said about the last frames
    - including the ones it refused.
    """
    here = pathlib.Path(os.environ.get("NR_ROOT") or pathlib.Path(__file__).resolve().parents[2])
    print(f"  version   {_version(here)}")
    status()
    log = nr_paths.LOG
    if not log.exists():
        print(f"  log       {log} - nothing yet; the daemon writes it when started by these tools")
        return
    try:
        text = log.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as error:
        print(f"  log       {log} - could not read it: {error}")
        return
    # Logs append across restarts. Do not combine an old device/config with new frames.
    starts = [i for i, line in enumerate(text) if line.startswith("model ready")]
    if starts:
        text = text[starts[-1] :]
    device = [l for l in text if l.startswith(("model ready", "buffers in", "runtime options"))]
    frames = [l for l in text if FIELDS_EXTENT.match(l)]
    refused = [l for l in text if "rejected/failed" in l or "GPU lost" in l]
    print("  log       %s  (%d frames, %d refused)" % (log, len(frames), len(refused)))
    for line in device:
        print(f"    {line}")
    for line in frames[-lines:]:
        print(f"    {line}")
    if refused:
        print("  refused, most recent last:")
        for line in refused[-3:]:
            print(f"    {line[:160]}")


def apply(knob, value):
    knob = ALIAS.get(knob, knob)
    if knob not in KNOBS:
        print("  no knob {!r}; try: {}".format(knob, ", ".join(KNOBS)))
        return
    entry = nr_knobs.BY_NAME[knob]
    if entry.kind == "choice":
        if value not in PROFILES:
            print("  profiles are: {}".format(", ".join(PROFILES)))
            return
    else:
        try:
            value = float(value)
        except ValueError:
            print(f"  {knob} takes a number")
            return
        if not entry.low <= value <= entry.high:
            print(f"  {knob} is between {entry.low:g} and {entry.high:g}")
            return
    values = read()
    values[knob] = value
    write(values)
    extra = ""
    if knob == "render_scale":
        extra = "   use report for measured frame times and network extent"
    print(f"  {knob} = {value}{extra}")
    if knob in ("detail_strength", "colour_strength") and value != 1.0:
        print("     note: away from 1 this adds a full-frame blur; check the daemon's")
        print(
            "     measured frame times. OpenCV is %s in this interpreter."
            % ("available" if _opencv() else "not available")
        )


def explain(name):
    """What a knob does, from the one table the panel and the manual also render."""
    entry = nr_knobs.BY_NAME.get(name)
    if entry is None:
        print("  no knob {!r}; try: {}".format(name, ", ".join(KNOBS)))
        return
    if entry.kind == "choice":
        span = " | ".join(PROFILES)
    else:
        span = f"{entry.low:g} to {entry.high:g}, step {entry.step:g}, daemon default {entry.default:g}"
    print(f"  {entry.label} - {entry.summary}")
    print(f"  {span}\n")
    for line in textwrap.wrap(entry.detail.replace("*", ""), 76):
        print("  " + line)


def command(words):
    if not words:
        return True
    head, rest = words[0].lower(), words[1:]
    if head in ("q", "quit", "exit"):
        return False
    if head == "on":
        TRIGGER.parent.mkdir(parents=True, exist_ok=True)
        TRIGGER.touch()
        print("  effect ON")
    elif head == "off":
        TRIGGER.unlink(missing_ok=True)
        print("  effect off")
    elif head in ("status", "st", ""):
        status()
    elif head == "set" and len(rest) == 2:
        apply(rest[0], rest[1])
    elif head in ALIAS or head in KNOBS:
        if len(rest) != 1:
            print(f"  {head} <value>")
        else:
            apply(head, rest[0])
    elif head in ("report", "bug"):
        report()
    elif head in ("rates", "fps"):
        print(
            f"  Arc 140V reference, whole round trip ({nr_knobs.RATES_MEASURED}); not a prediction for this GPU:\n"
        )
        print("    %-14s %8s %9s %8s" % ("swapchain", "scale", "ms", "fps"))
        for width, height, scale, ms in RATES:
            print(
                "    %-14s %8.2f %9.1f %8.2f" % ("%dx%d" % (width, height), scale, ms, 1000.0 / ms)
            )
        print("\n  Set the *game* to the swapchain size; the compositor stretches the")
        print("  result to the panel for nothing. This is the daemon's own cost, with")
        print("  nothing else on the GPU; a game adds its own frame to it.")
    elif head in ("help", "?", "man"):
        if rest:
            explain(ALIAS.get(rest[0], rest[0]))
        else:
            print("  on | off | status | report | rates | set <knob> <value> | help <knob> | q")
            print("  shorthand: p <profile>, i <n>, d <n>, c <n>, s <scale>, t <0-1>")
            print("  knobs: {}".format(", ".join(KNOBS)))
            print("  `help <knob>` says what one actually does, and what it costs.")
    else:
        print("  ? {}   (try `help`)".format(" ".join(words)))
    return True


def main():
    if len(sys.argv) > 1:
        command(sys.argv[1:])
        return 0
    print("nr-ctl - `help` for commands, `q` to leave. The daemon and the game keep running.\n")
    status()
    while True:
        try:
            line = input("\nnr> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not command(line.split()):
            return 0


if __name__ == "__main__":
    sys.exit(main())
