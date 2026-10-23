"""
run_experiment.py -- FROZEN (karpathy-vanilla). The one way an agent runs experiments (the `ar-exp` command).

  ar-exp NAME=TRAIN_FILE [NAME=TRAIN_FILE ...]

For each experiment (at most the run's batch size per call; they run in parallel when the machine has the
GPUs; run settings are kept outside the agent's folder, in $AR_WORK/state/<arm>/<run-id>.json):
  1. snapshot TRAIN_FILE and the idea.md next to it into experiments/NAME/ (the agent cannot change them
     afterwards);
  2. run it on this machine's GPUs (8, or all of them if fewer), queued by common/localgpu.py
     (harness/run_vanilla.sh: torchrun train.py next to a fresh copy of the harness's prepare.py), with a
     time limit of TIME_LIMIT. There is no check before training;
  3. the score is what train.py prints (`val_bpb: ...` etc., as in Karpathy's autoresearch);
  4. writes experiments/NAME/result.json and run.log, appends one row to results.tsv, and makes one git
     commit for the experiment.
Blocks until all experiments of the call have finished, then prints a table of their results.
Stopping ar-exp cancels its queued and running trainings.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = Path(os.environ.get("AR_REPO") or HARNESS.parents[1])
WORK = Path(os.environ.get("AR_WORK") or REPO / "work")
sys.path.insert(0, str(REPO / "common"))
import localgpu  # noqa: E402  (the local GPU task queue)

ARMS = ("vanilla", "free")
STATE = WORK / "state"                               # run settings, <arm>/<run-id>.json (written by autoresearch.py)
RAW = WORK / "raw_logs"                              # full job logs, <arm>/<run-id>/..., never shown to the agent
OUTPUTS = WORK / "outputs"                           # the agents' run folders: <date>/<tag>/
TIME_LIMIT = "60m"                                   # per experiment (the only limit on a run)
TRAIN_GPUS = 8                                       # GPUs per training (fewer if the machine has fewer)
SUBMITTED = []                                       # this call's GPU tasks
COLS = ["name", "status", "val_bpb", "training_seconds", "peak_vram_mb", "num_steps", "num_params_M", "finished", "idea"]


def require_pool():
    """GPUs must be visible: agents' calls reach this through the launcher's relay, outside their jail"""
    if not localgpu.available():
        sys.exit("no GPUs are visible from here (this command runs through the experiment relay)")


def find_state(run_dir):
    """(arm, settings file) of the run whose folder is run_dir"""
    for arm in ARMS:
        f = STATE / arm / f"{run_dir.parent.name}__{run_dir.name}.json"
        if f.exists():
            return arm, f
    return None, None


def fail(msg):
    sys.exit(f"ar-exp: {msg}")


def git(run_dir, *args):
    return subprocess.run(["git", "-C", str(run_dir), "-c", "user.name=autoresearch",
                           "-c", "user.email=autoresearch@local", *args], capture_output=True, text=True)


def first_line(path):
    try:
        return Path(path).read_text().strip().splitlines()[0].lstrip("# ").strip()[:150]
    except (FileNotFoundError, IndexError):
        return ""


def stop(signum, _frame):
    """a stopped ar-exp must not leave its runs going on the pool, unrecorded"""
    for tid in SUBMITTED:
        rec = localgpu.get(tid)
        if not rec or rec["state"] not in localgpu.FINAL:
            localgpu.cancel(tid)
    print("ar-exp: stopped; its unfinished runs were cancelled", flush=True)
    os._exit(128 + signum)


def wait(tid, poll=10):
    while True:
        rec = localgpu.get(tid)
        if rec and rec["state"] in localgpu.FINAL:
            return rec
        time.sleep(poll)


def train_gpus():
    return min(TRAIN_GPUS, localgpu.num_gpus())


def run_one(run_dir, arm, name, train_src):
    exp = run_dir / "experiments" / name
    raw = RAW / arm / f"{run_dir.parent.name}__{run_dir.name}" / name
    exp.mkdir(parents=True)
    raw.mkdir(parents=True, exist_ok=True)
    train = exp / "train.py"
    shutil.copy(train_src, train)
    idea = Path(train_src).parent / "idea.md"
    if idea.exists():
        shutil.copy(idea, exp / "idea.md")
    out = dict(name=name, idea=first_line(exp / "idea.md"))
    meta = dict(submitted=time.time(), harness_commit=subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
    tdir = raw / "run"
    tdir.mkdir(exist_ok=True)
    tid = localgpu.submit(f"bash {HARNESS / 'run_vanilla.sh'} {train} {tdir}", gpus=train_gpus(), time=TIME_LIMIT,
                    project="autoresearch", workdir=str(tdir), log=str(tdir / "task.log"),
                    name=f"{run_dir.name}-{name}")
    SUBMITTED.append(tid)
    meta["task"] = tid
    rec = wait(tid)
    res = json.loads((tdir / "result.json").read_text()) if (tdir / "result.json").exists() else {}
    if (tdir / "run.log").exists():
        log = (tdir / "run.log").read_text(errors="replace").replace("\r", "\n")
        lines = [l for l in log.splitlines() if l.strip()]
        (exp / "run.log").write_text("\n".join(lines[-400:]) + "\n")   # what the agent reads (last 400 lines)
    if rec["state"] == "timeout":
        out.update(status="timeout", reason=f"stopped at the {TIME_LIMIT} limit")
    elif "val_bpb" in res:
        out.update(status="trained")
    else:
        out.update(status="crash", reason="no val_bpb printed; see run.log")
    out.update({k: res[k] for k in ("val_bpb", "training_seconds", "peak_vram_mb", "num_steps", "num_params_M") if k in res})
    return out, meta


def main(argv):
    require_pool()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    run_dir = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                                  text=True).stdout.strip() or ".").resolve()
    arm, state = find_state(run_dir)
    if run_dir.parent.parent != OUTPUTS.resolve() or not state:
        fail("run this from your run folder")
    cfg = json.loads(state.read_text())
    if not argv or any("=" not in a for a in argv):
        fail("usage: ar-exp NAME=TRAIN_FILE [NAME=TRAIN_FILE ...]")
    if len(argv) > cfg["batch"]:
        fail(f"at most {cfg['batch']} experiment(s) per call in this run")
    tsv = run_dir / "results.tsv"
    used = max(0, len(tsv.read_text().splitlines()) - 2) if tsv.exists() else 0
    if cfg.get("budget") and used + len(argv) > cfg["budget"]:
        fail(f"experiment budget: {used} of {cfg['budget']} used; {cfg['budget'] - used} left, this call asks for "
             f"{len(argv)}" if used < cfg["budget"] else
             f"experiment budget used up ({used} of {cfg['budget']}): finish notes.md and end your turn")
    jobs = []
    for a in argv:
        name, src = a.split("=", 1)
        if not name.replace("-", "").replace("_", "").isalnum():
            fail(f"experiment name {name!r}: use letters, digits, - and _")
        if (run_dir / "experiments" / name).exists() or name in [j[0] for j in jobs]:
            fail(f"experiment {name!r} already exists; pick a new name")
        if not Path(src).is_file():
            fail(f"{src} is not a file")
        jobs.append((name, Path(src).resolve()))
    print(f"ar-exp: {len(jobs)} experiment(s): train.py on {train_gpus()} GPUs each (limit {TIME_LIMIT}); "
          "results follow", flush=True)
    with ThreadPoolExecutor(len(jobs)) as ex:
        pairs = list(ex.map(lambda j: run_one(run_dir, arm, *j), jobs))
    results = [r for r, _ in pairs]

    for r in results:
        r["finished"] = time.strftime("%Y-%m-%d %H:%M")
        (run_dir / "experiments" / r["name"] / "result.json").write_text(json.dumps(r, indent=2) + "\n")
        meta = next(m for rr, m in pairs if rr is r)
        (RAW / arm / f"{run_dir.parent.name}__{run_dir.name}" / r["name"] / "meta.json").write_text(
            json.dumps({**r, **meta}, indent=2) + "\n")
        row = {**r, "val_bpb": f"{r['val_bpb']:.6f}" if isinstance(r.get("val_bpb"), float) else r.get("val_bpb", "-")}
        with tsv.open("a") as f:
            f.write("\t".join(str(row.get(c) if row.get(c) is not None else "-") for c in COLS) + "\n")
        git(run_dir, "add", "-A", "--", ".", ":(exclude)trace")
        git(run_dir, "commit", "-q", "--no-verify", "-m", f"{r['name']}: {r['status']} {row['val_bpb']}  {r['idea'][:100]}")
    print(f"\n{'name':24} {'status':10} {'val_bpb':>9} {'train_s':>8} {'steps':>6}  idea / reason")
    for r in results:
        v = f"{r['val_bpb']:.6f}" if isinstance(r.get("val_bpb"), float) else "-"
        print(f"{r['name']:24} {r['status']:10} {v:>9} {r.get('training_seconds', '-'):>8} "
              f"{r.get('num_steps', '-'):>6}  {r.get('reason') or r['idea']}")
    print(f"\nbaseline: {cfg['baseline_val']:.6f} (lower is better). All results: results.tsv; details: experiments/<name>/")
    if cfg.get("budget"):
        used = len(tsv.read_text().splitlines()) - 2
        print(f"experiment budget: {used} of {cfg['budget']} used, {cfg['budget'] - used} left")


if __name__ == "__main__":
    main(sys.argv[1:])
