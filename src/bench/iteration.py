"""One optimisation iteration -> one structured record, automatically.

    uv run --active python src/bench/iteration.py LABEL [--skip-tests]

Runs the correctness suites, the sustained GEMM and frame benches (2 s each, they
append their own lines to notes/perf-log.jsonl under LABEL) with XMX_PIPELINE_STATS
on, then appends ONE JSON line to notes/runs.jsonl: commit, tests (pass/fail + tail
of failures), per-pipeline spill/fill/scratch/live/slm, every GEMM shape and the
frame breakdown. Also prints that record. Build first (make); this does not build.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["test_staged_partial", "test_staged32", "test_staged_packed", "test_gemm_residual",
         "test_gemm_qkv", "test_window_attention", "test_window_block", "test_global_attention",
         "test_ffn_fused", "test_portable", "test_resident", "test_frame_execution"]


def run(cmd, env=None, timeout=1800):
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def spills(tsv):
    out = []
    if not tsv.exists():
        return out
    keys = {"flags": "flags", "subgroup": "subgroup", "spill": "Spill Count",
            "fill": "Fill Count", "scratch": "Scratch Memory Size", "live": "Max live registers",
            "slm": "Workgroup Memory Size", "cycles": "Cycle Count",
            "instructions": "Instruction Count"}
    for line in tsv.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        rest = "\t".join(f[1:])
        rec = {"pipeline": f[0]}
        for k, name in keys.items():
            m = re.search(re.escape(name) + r"=(-?\d+)", rest)
            rec[k] = int(m.group(1)) if m else None
        out.append(rec)
    return out


def bench_lines(label, kind):
    log = ROOT / "notes" / "perf-log.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    return [r for r in rows if r.get("label") == label and r.get("kind", "frame") == kind]


def compare(labels):
    """Table of every recorded label (or the given ones) vs the first: frame ms + GEMM TFLOP/s."""
    log = ROOT / "notes" / "perf-log.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    labels = labels or list(dict.fromkeys(r["label"] for r in rows))
    shapes = list(dict.fromkeys(r["shape"] if isinstance(r.get("shape"), str) else str(r.get("shape"))
                                for r in rows if r.get("kind") == "gemm"))
    print("%-22s" % "metric" + "".join("%14s" % l[:13] for l in labels))
    for key in ("wall_ms", "device_ms", "fps"):
        vals = [(bench_lines(l, "frame") or [{}])[-1].get(key) for l in labels]
        print("%-22s" % ("frame " + key) + "".join("%14s" % v for v in vals))
    for s in shapes:
        vals = []
        for l in labels:
            g = [r for r in bench_lines(l, "gemm") if str(r.get("shape")) == s]
            vals.append(g[-1].get("tflops") if g else None)
        print("%-22s" % s[:22] + "".join("%14s" % v for v in vals))


def summary(label):
    """The latest run of LABEL from notes/runs.jsonl: tests, frame passes, worst spills."""
    runs = [json.loads(l) for l in (ROOT / "notes" / "runs.jsonl").read_text().splitlines()
            if l.strip()]
    rec = [r for r in runs if r["label"] == label][-1]
    print("%s  %s  env %s" % (label, rec["time"], " ".join(rec.get("env", [])) or "-"))
    print("tests %d/%d%s" % (rec["tests_passed"], rec["tests_total"],
          "".join("  FAIL " + t for t, v in rec["tests"].items() if not v["pass"])))
    frame = rec.get("frame") or {}
    print("frame wall %s ms  device %s ms  %s fps" % (frame.get("wall_ms"),
          frame.get("device_ms"), frame.get("fps")))
    for name, ms in (frame.get("passes") or {}).items():
        print("  %-28s %8.2f ms" % (name, ms))
    worst = {}
    for p in rec["pipelines"]:
        w = worst.setdefault(p["pipeline"], p)
        if (p["spill"] or 0) > (w["spill"] or 0):
            worst[p["pipeline"]] = p
    print("spills (worst specialisation)  total %d" % rec["spill_total"])
    for name, p in sorted(worst.items(), key=lambda kv: -(kv[1]["spill"] or 0)):
        print("  %-30s sg %-3s spill %-5s fill %-5s live %-4s slm %s" % (name, p["subgroup"],
              p["spill"], p["fill"], p["live"], p["slm"]))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        return compare(sys.argv[2:])
    if len(sys.argv) > 2 and sys.argv[1] == "--summary":
        return summary(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--seconds", default="0.5")   # warm-up; the timed phase is --runs frames
    ap.add_argument("--runs", default="12")
    a = ap.parse_args()
    py = [sys.executable]
    rec = {"label": a.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "commit": run(["git", "rev-parse", "--short", "HEAD"])[1].strip(),
           "dirty": bool(run(["git", "status", "--porcelain", "src"])[1].strip())}

    out_dir = ROOT / "notes" / "runs" / a.label        # every raw output of the run
    out_dir.mkdir(parents=True, exist_ok=True)
    env_file = out_dir / "env.txt"
    rec["env"] = env_file.read_text().split() if env_file.exists() else []

    rec["tests"] = {}
    if not a.skip_tests:
        for t in TESTS:
            code, out = run(py + [f"src/gpu/{t}.py"], timeout=900)
            (out_dir / f"{t}.log").write_text(out)
            rec["tests"][t] = {"pass": code == 0} | ({} if code == 0 else {"tail": out[-800:]})
    rec["tests_passed"] = sum(v["pass"] for v in rec["tests"].values())
    rec["tests_total"] = len(rec["tests"])

    code, out = run(py + ["src/bench/gemm_sustained.py", "--record", a.label])
    (out_dir / "gemm_sustained.log").write_text(out)
    rec["gemm_exit"] = code
    tsv = out_dir / "pipeline-stats.tsv"
    tsv.unlink(missing_ok=True)
    env = dict(os.environ, XMX_PIPELINE_STATS=str(tsv))
    code, out = run(py + ["src/bench/frame_profile.py", "--size", "576", "1024",
                          "--seconds", a.seconds, "--runs", a.runs, "--record", a.label], env=env)
    (out_dir / "frame_profile.log").write_text(out)
    rec["frame_exit"] = code
    rec["gemm"] = bench_lines(a.label, "gemm")
    frame = bench_lines(a.label, "frame")
    rec["frame"] = frame[-1] if frame else None
    rec["pipelines"] = spills(tsv)
    rec["spill_total"] = sum(p["spill"] or 0 for p in rec["pipelines"])

    with open(ROOT / "notes" / "runs.jsonl", "a") as sink:
        sink.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))
    (out_dir / "record.json").write_text(json.dumps(rec, indent=1))
    sys.exit(0 if rec["tests_passed"] == rec["tests_total"] and not rec["gemm_exit"]
             and not rec["frame_exit"] else 1)


if __name__ == "__main__":
    main()
