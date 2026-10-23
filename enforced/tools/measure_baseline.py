#!/usr/bin/env python3
"""
measure_baseline.py -- measure harness/baseline_model.py on this machine's GPUs (common/localgpu.py): an LR
sweep (full trainings) and repeated speed benchmarks (the ar-check benchmark), plus one model check for
sizes and FLOPs. Writes OUT/summary.json and prints a table; it does not change
baseline/baseline.json or harness/caps.json (that is done, and committed, after reviewing the numbers).

  python tools/measure_baseline.py --out baseline/measure_2026-10-07 [--lrs 1e-3,3.16e-3,1e-2,3.16e-2,1e-1]
         [--speed-runs 24] [--seeds 3] [--priority 100]

--seeds N: after the sweep, N more trainings at the best LR with seeds 1..N (the sweep's run is seed 1337).

All tasks are submitted at once with --priority (ahead of other queued GPU tasks). On hardware other than
8 H100s, the step-time cap in harness/caps.json (baseline step time x 1.05) must come from this measurement:
the cap we shipped was measured on 8 H100s.
"""

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

MASTER = Path(__file__).resolve().parents[1]
HARNESS = MASTER / "harness"
sys.path.insert(0, str(HARNESS))
import paths  # noqa: E402
import localgpu  # noqa: E402

ENV = f"source {HARNESS}/env.sh; export TMPDIR=/tmp/$USER-$GQ_TASK_ID; mkdir -p $TMPDIR"


def submit(cmd, gpus, limit, d, name, prio):
    d.mkdir(parents=True, exist_ok=True)
    tid = localgpu.submit(cmd, gpus=gpus, time=limit, project="autoresearch", workdir=str(d), log=str(d / "task.log"),
                          name=name, env=ENV if gpus == 1 else None)
    localgpu.set_priority(tid, prio)
    return tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", default="1e-3,3.16e-3,1e-2,3.16e-2,1e-1")
    ap.add_argument("--speed-runs", type=int, default=24)
    ap.add_argument("--priority", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=0)
    a = ap.parse_args()
    out = (MASTER / a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = out / "model.py"
    shutil.copy(HARNESS / "baseline_model.py", model)
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    commit = __import__("subprocess").run(["git", "-C", str(MASTER), "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip()
    tasks = {"check": submit(f"python {HARNESS / 'smoke.py'} {model} --json {out / 'check' / 'check.json'}", 1, "20m",
                             out / "check", "baseline-check", a.priority)}
    for lr in a.lrs.split(","):
        tasks[f"lr_{lr}"] = submit(f"bash {HARNESS / 'run_train.sh'} {model} {lr} {out / f'lr_{lr}'} train",
                                   paths.train_gpus(), "40m", out / f"lr_{lr}", f"baseline-lr-{lr}", a.priority)
    for i in range(a.speed_runs):
        d = out / f"speed_{i:02d}"
        tasks[f"speed_{i:02d}"] = submit(f"bash {HARNESS / 'run_train.sh'} {model} 1e-2 {d} speed",
                                         paths.train_gpus(), "15m", d, f"baseline-speed-{i:02d}", a.priority)
    (out / "tasks.json").write_text(json.dumps(dict(model_sha256=sha, commit=commit, tasks=tasks), indent=2) + "\n")
    print(f"submitted {len(tasks)} tasks (priority {a.priority}); waiting ...", flush=True)
    recs = {}
    while len(recs) < len(tasks):
        for k, tid in tasks.items():
            if k not in recs:
                r = localgpu.get(tid)
                if r and r["state"] in localgpu.FINAL:
                    recs[k] = r
                    print(f"[{time.strftime('%H:%M:%S')}] {k}: {r['state']} on {r.get('host')}", flush=True)
        time.sleep(20)

    def wait_all(keys):
        while any(k not in recs for k in keys):
            for k in keys:
                if k not in recs:
                    r = localgpu.get(tasks[k])
                    if r and r["state"] in localgpu.FINAL:
                        recs[k] = r
                        print(f"[{time.strftime('%H:%M:%S')}] {k}: {r['state']} on {r.get('host')}", flush=True)
            time.sleep(20)

    res = lambda k: json.loads((out / k / "result.json").read_text()) if (out / k / "result.json").exists() else None
    check = json.loads((out / "check" / "check.json").read_text()) if (out / "check" / "check.json").exists() else {}
    sweep = {}
    for k in tasks:
        if k.startswith("lr_"):
            r = res(k) or {}
            log = (out / k / "train.log").read_text() if (out / k / "train.log").exists() else ""
            vals = re.findall(r"step:(\d+)/\d+ val_loss:([\d.]+)", log)
            sweep[k[3:]] = dict(status=r.get("status"), val_loss=r.get("val_loss"), step_ms=r.get("check_step_ms"),
                                step_avg_ms=r.get("step_avg_ms"), host=recs[k].get("host"),
                                val_curve=[(int(s), float(v)) for s, v in vals])
    speed = [dict(run=k, host=recs[k].get("host"), ms=(res(k) or {}).get("check_step_ms")) for k in tasks
             if k.startswith("speed_")]
    ms = [s["ms"] for s in speed if s["ms"]]
    by_host = {}
    for s in speed:
        if s["ms"]:
            by_host.setdefault(s["host"], []).append(s["ms"])
    ok = {lr: v for lr, v in sweep.items() if v["val_loss"]}
    best = min(ok, key=lambda lr: ok[lr]["val_loss"]) if ok else None
    seeds = {}
    if a.seeds and best:
        keys = []
        for sd in range(1, a.seeds + 1):
            k = f"seed_{sd}"
            tasks[k] = submit(f"bash {HARNESS / 'run_train.sh'} {model} {best} {out / k} train {sd}",
                              paths.train_gpus(), "40m", out / k, f"baseline-seed-{sd}", a.priority)
            keys.append(k)
        print(f"best LR {best}: {ok[best]['val_loss']}; submitted {a.seeds} seed runs", flush=True)
        wait_all(keys)
        seeds = {"1337": ok[best]["val_loss"], **{k[5:]: (res(k) or {}).get("val_loss") for k in keys}}
        (out / "tasks.json").write_text(json.dumps(dict(model_sha256=sha, commit=commit, tasks=tasks), indent=2) + "\n")
    vals = [v for v in seeds.values() if v]
    summary = dict(model_sha256=sha, commit=commit, check=check, lr_sweep=sweep, best_lr=best,
                   best_val=ok[best]["val_loss"] if best else None, seeds=seeds,
                   seed_mean=statistics.mean(vals) if vals else None,
                   seed_sd=statistics.stdev(vals) if len(vals) > 1 else None, speed_runs=speed,
                   speed_stats=dict(n=len(ms), mean=statistics.mean(ms) if ms else None,
                                    stdev=statistics.stdev(ms) if len(ms) > 1 else None,
                                    min=min(ms) if ms else None, max=max(ms) if ms else None,
                                    by_host={h: dict(n=len(v), mean=statistics.mean(v), min=min(v), max=max(v))
                                             for h, v in sorted(by_host.items())}))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nLR sweep:")
    for lr, v in sweep.items():
        print(f"  {lr:>8}  {v['status']!s:10} val {v['val_loss'] or '-'}  step {v['step_ms'] or '-'} ms  ({v['host']})")
    print(f"best LR {best}: {summary['best_val']}")
    if vals:
        print(f"seeds at {best}: {seeds}; mean {summary['seed_mean']:.5f}, sd {summary['seed_sd'] or 0:.5f}")
    st = summary["speed_stats"]
    if ms:
        print(f"\nspeed ({st['n']} runs): mean {st['mean']:.1f} sd {st['stdev'] or 0:.1f} min {st['min']:.1f} max {st['max']:.1f} ms")
        for h, v in st["by_host"].items():
            print(f"  {h:14} n={v['n']}  mean {v['mean']:.1f}  min {v['min']:.1f}  max {v['max']:.1f}")
    print(f"\nparams {check.get('total_params')}, components {check.get('components')}, flops/token {check.get('flops_per_token')}")
    print(f"summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
