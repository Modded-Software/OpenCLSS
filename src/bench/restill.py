#!/usr/bin/env python3
"""Replay a `--dump` capture through today's daemon, frame by frame, in the order it was taken.

`nr_daemon.py --dump DIR` keeps what the game handed it (`NNN_in.png`) and what it sent back
(`NNN_out.png`). Sent to a fresh daemon in the same order, the `NNN_in` frames are the same
input - and, since the history is built only from frames the daemon saw, the same temporal
state - through the graph as it is now. So a published still can be re-rendered on the frame
the owner chose, without starting the game again.

    python3 src/bench/restill.py work/tekken work/restill/tekken
    python3 src/bench/restill.py work/mk1cap work/restill/mk1 --set profile=natural

The answers land in OUT through the daemon's own `--dump`, numbered from 001 in the order sent,
and each `NNN_in` it writes is checked against the frame it was sent. `render_scale` is 1.0 unless set:
the published stills are taken with the model at full resolution. Every swapchain behind the
captures so far was B8G8R8A8_UNORM, and the dumps are 8-bit, so the bytes sent are the bytes
the game sent.

`--native` replays through the C frame library instead - `nr_frame --replay`, which runs the
daemon's frame through `nr_frame_live` in one process: the same letterbox, render extent and
history, no socket, and on whichever runtime `NR_GPU_BACKEND` names (Vulkan, Metal, Direct3D
12), so a capture can be re-rendered where the daemon and its Unix socket do not run. The
answers match the daemon's to the rounding of the noise channels, which NumPy and C compute
to a last bit apart (`notes/phase68`); `src/ref/test_nr_frame_live.py` holds the rest of it
byte for byte.

    python3 src/bench/restill.py work/tekken work/restill/tekken --native
"""
import argparse
import json
import pathlib
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "ref"))
sys.path.insert(0, str(ROOT / "src"))
import image_io
import nr_build

MAGIC = 0x304E524E
FORMAT_B8G8R8A8 = 44


def ffmpeg():
    """The ffmpeg to encode with: on PATH, else the BtbN build the video tools use."""
    return shutil.which("ffmpeg") or str(pathlib.Path.home() / ".local" / "bin" / "ffmpeg")


def stitch(out, fps):
    """Stitch the replay's own PNGs into videos so the result can be watched.

    OUT/rendered.mp4 is the answers alone; OUT/compare.mp4 is `NNN_in` beside `NNN_out`,
    left to right, the game's frame against the model's, so the two can be judged together.
    Both use the same frame count and fps, so `rendered.mp4` is the thing to watch and
    `compare.mp4` is what to freeze on when a frame looks wrong.
    """
    exe = ffmpeg()
    outs = sorted(out.glob("*_out.png"))
    if not outs:
        return
    seconds = len(outs) / fps
    rendered = out / "rendered.mp4"
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(out / "%03d_out.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "16", str(rendered)], check=True)
    print(f"  {len(outs)} frames, {seconds:.1f}s -> {rendered}", flush=True)
    if list(out.glob("*_in.png")):
        compare = out / "compare.mp4"
        subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(out / "%03d_in.png"), "-framerate", str(fps),
                        "-i", str(out / "%03d_out.png"),
                        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "16", str(compare)], check=True)
        print(f"  input | rendered side by side -> {compare}", flush=True)


def frames_of(directory):
    # numbered as the daemon numbers them; anything else in the folder is somebody's own
    frames = sorted((path for path in directory.glob("*_in.png") if path.stem.split("_")[0].isdigit()),
                    key=lambda path: int(path.stem.split("_")[0]))
    numbers = [int(path.stem.split("_")[0]) for path in frames]
    if not frames:
        raise SystemExit(f"no NNN_in.png in {directory}")
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        # a gap means frames the daemon saw and this replay would not: its history differs
        raise SystemExit(f"{directory}: the frames are not consecutive ({numbers[0]}-{numbers[-1]}, "
                         f"{len(numbers)} of them)")
    return frames


def bgra(path):
    rgb = np.rint(image_io.load(path) * 255.0).astype(np.uint8)
    height, width = rgb.shape[:2]
    out = np.empty((height, width, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = rgb[..., 2], rgb[..., 1], rgb[..., 0]
    out[..., 3] = 255
    return width, height, rgb, out.tobytes()


def exchange(path, width, height, payload):
    want = 4 * width * height
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(600)
        client.connect(path)
        client.sendall(struct.pack("<4I", MAGIC, width, height, FORMAT_B8G8R8A8) + payload)
        got = 0
        while got < want:
            piece = client.recv(want - got)
            if not piece:
                raise RuntimeError(f"the daemon answered {got} of {want} bytes; its log says why")
            got += len(piece)


# the daemon's settings file names, as `nr_frame --replay` spells them
NATIVE_FLAGS = {"render_scale": "--render-scale", "profile": "--profile", "intensity": "--intensity",
                "detail_strength": "--detail-strength", "colour_strength": "--colour-strength",
                "temporal": "--temporal", "hold": "--hold", "release": "--release",
                "cut_limit": "--cut-limit", "min_extent": "--min-extent"}


def git_revision():
    """The commit this run came from, for provenance. Surfaces git's own words if it cannot say."""
    done = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        print(f"  git rev-parse failed ({done.returncode}): {done.stderr.strip()}", flush=True)
        return "unknown"
    revision = done.stdout.strip()
    if not revision:
        print("  git rev-parse printed no revision", flush=True)
        return "unknown"
    return revision


def frame_times(log_text):
    """The per-frame graph times the replay logged, in ms, and the frame rate they imply.

    `nr_frame` prints `... in 0.10s ...` a frame, wrapping only the graph run: PNG read and
    write sit outside the timer, so the rate here is the model's, not the replay's. The
    summary line is preferred when the binary prints one; otherwise it is computed from the
    per-frame lines, which every build prints."""
    times = [1000.0 * float(m) for m in re.findall(r"\d+x\d+ in (\d+\.\d+)s ", log_text)]
    return times


def statistics_of(times, summary):
    if summary:
        values = {}
        for token, key in (("mean", "mean_ms"), ("median", "median_ms"),
                           ("min", "min_ms"), ("p95", "p95_ms"), ("max", "max_ms")):
            found = re.search(rf"{key.split('_')[0]} (\d+\.\d+)", summary)
            if found:
                values[key] = float(found.group(1))
        return values
    if not times:
        return {}
    ordered = sorted(times)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    return {"mean_ms": sum(ordered) / n, "median_ms": median, "min_ms": ordered[0],
            "p95_ms": ordered[int(0.95 * (n - 1))], "max_ms": ordered[-1]}


def write_report(out, source, settings, log_text, seconds):
    """The run's metrics and where it came from, in one file beside the frames it describes.

    OUT/replay.log is the replay's whole log; OUT/report.json is the same run's numbers, so a
    number quoted later has a program and a revision behind it. `frame_rate_fps` is the
    model's: the tool times the graph alone, PNG read and write excluded."""
    (out / "replay.log").write_text(log_text)
    summary = [line.strip() for line in log_text.splitlines() if line.strip().startswith("summary")]
    summary = summary[0] if summary else None
    times = frame_times(log_text)
    stats = statistics_of(times, summary)
    fps = 1e3 / stats["median_ms"] if stats.get("median_ms") else None
    report = {
        "source": str(source), "out": str(out),
        "frames": len(list(out.glob("*_out.png"))), "video_seconds": seconds,
        "frame_rate_fps": round(fps, 2) if fps else None,
        "graph_time_excludes_png_io": True,
        "timing_ms": {k: round(v, 3) for k, v in stats.items()},
        "settings": settings, "git_revision": git_revision(),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "videos": [name for name in ("rendered.mp4", "compare.mp4") if (out / name).exists()],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"  {out / 'report.json'}", flush=True)
    if fps:
        print(f"  {len(times)} frames  median {stats['median_ms']:.2f} ms  "
              f"mean {stats['mean_ms']:.2f} ms  ->  {fps:.2f} fps  (graph, excludes PNG I/O)",
              flush=True)
    return report


def native(frames, source, out, settings):
    """The replay through `nr_frame --replay`, then the same check the daemon's gets."""
    command = [str(nr_build.executable("nr_frame")), "--replay", str(source), str(out)]
    for knob, value in settings.items():
        if knob not in NATIVE_FLAGS:
            raise SystemExit(f"--native has no {knob}; it takes " + ", ".join(sorted(NATIVE_FLAGS)))
        command += [NATIVE_FLAGS[knob], str(int(value) if knob == "min_extent" else value)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit("nr_frame --replay failed:\n" + done.stdout + done.stderr)
    for index, path in enumerate(frames, 1):
        written = out / f"{index:03d}_in.png"
        try:
            same = np.array_equal(np.rint(image_io.load(written) * 255.0), np.rint(image_io.load(path) * 255.0))
        except RuntimeError as error:
            # No ImageMagick: the frames are on disk and the replay logged each one, but the
            # pixels cannot be compared here. Say so once; do not fail a run that succeeded.
            print(f"  cannot compare {written} to {path}: {error}", flush=True)
            return done.stdout
        if not same:
            raise SystemExit(f"{written} is not the frame sent from {path}")
    print(f"  every NNN_in in {out} is the frame it was sent", flush=True)
    return done.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=pathlib.Path, help="a --dump directory")
    parser.add_argument("out", type=pathlib.Path, help="an empty or new directory")
    parser.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE",
                        help="a daemon setting, e.g. render_scale=0.9 or profile=natural")
    parser.add_argument("--native", action="store_true",
                        help="through the C frame library (nr_frame --replay), not the daemon")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="the frame rate the replay's PNGs are stitched back at (default 30)")
    parser.add_argument("--no-video", action="store_true",
                        help="leave only the PNGs; no rendered.mp4 / compare.mp4")
    args = parser.parse_args()
    settings = {"render_scale": 1.0}
    for item in args.set:
        knob, _, value = item.partition("=")
        try:
            settings[knob] = float(value)
        except ValueError:
            settings[knob] = value
    frames = frames_of(args.source)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty: the daemon numbers its dump after what is there")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.native:
        log_text = native(frames, args.source, args.out, settings)
        if not args.no_video:
            stitch(args.out, args.fps)
        write_report(args.out, args.source, settings, log_text, len(frames) / args.fps)
        return

    daemon_log = ""
    with tempfile.TemporaryDirectory() as directory:
        room = pathlib.Path(directory)
        daemon_socket, settings_file, log = room / "restill.sock", room / "settings.json", room / "log"
        settings_file.write_text(json.dumps(settings))
        with open(log, "w") as sink:
            daemon = subprocess.Popen(
                [sys.executable, str(ROOT / "src" / "layer" / "nr_daemon.py"),
                 "--socket", str(daemon_socket), "--settings", str(settings_file),
                 "--max-pixels", str(1920 * 1200), "--dump", str(args.out)],
                stdout=sink, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if daemon_socket.exists():
                    break
                if daemon.poll() is not None:
                    raise SystemExit("daemon exited:\n" + log.read_text())
                time.sleep(0.5)
            print(f"  {args.source} -> {args.out}, {len(frames)} frames, settings {settings}",
                  flush=True)
            for index, path in enumerate(frames, 1):
                width, height, rgb, payload = bgra(path)
                exchange(str(daemon_socket), width, height, payload)
                written = args.out / f"{index:03d}_in.png"
                # The dump is written after the answer is sent, and the frame's log line after
                # the dump: wait for the line, and both pictures are on disk.
                said = []
                for _ in range(600):
                    said = [line.strip() for line in log.read_text().splitlines() if "B8G8R8A8" in line]
                    if len(said) >= index or daemon.poll() is not None:
                        break
                    time.sleep(0.1)
                if len(said) < index:
                    raise SystemExit("the daemon did not log frame %d:\n%s" % (index, log.read_text()))
                same = np.array_equal(np.rint(image_io.load(written) * 255.0).astype(np.uint8), rgb)
                print(f"  {path.name} -> {written.name}  {'same frame' if same else 'DIFFERENT FRAME'}"
                      f"  {said[index - 1]}", flush=True)
                if not same:
                    raise SystemExit(f"{written} is not the frame sent from {path}")
        finally:
            daemon_log = log.read_text()
            daemon.terminate()
            daemon.wait(timeout=60)
    if not args.no_video:
        stitch(args.out, args.fps)
    write_report(args.out, args.source, settings, daemon_log, len(frames) / args.fps)


if __name__ == "__main__":
    main()
