"""
run_experiment.py -- FROZEN. The one way an agent runs experiments (installed as the `ar-exp` command).

  ar-exp NAME=MODEL_FILE [NAME=MODEL_FILE ...]

For each experiment, in parallel (at most the run's batch size per call; run settings are kept
outside the agent's folder, in $AR_WORK/state/enforced/<run-id>.json):
  1. snapshot MODEL_FILE and the idea.md next to it into experiments/NAME/ (the agent cannot change
     them afterwards);
  2. requires that this exact model (same code, ignoring comments and formatting) has passed ar-check
     (harness/check.py); ar-exp itself never checks or benchmarks. The stored check report becomes
     experiments/NAME/check.txt, and the benchmark's step time is the experiment's step_ms;
  3. training on 8 GPUs of the local GPU queue (harness/run_train.sh: 2.0B tokens, prepare.LR); the training
     measures its own step time over steps 20-120 and records it (train_step_ms), without enforcing it;
  4. writes experiments/NAME/result.json, appends one row to results.tsv, and makes one git commit
     for the experiment.
Blocks until all experiments of the call have finished, then prints a table of their results (on one
8-GPU machine the call's trainings run one after another).
A training that hangs (timeout) or loses its node is trained once more automatically.
Stopping ar-exp (Ctrl-C, a timeout, the agent's session ending) cancels its trainings.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanitize import model_error, train_curve  # noqa: E402
from prepare import LR  # noqa: E402
from check import check_record, model_hash, step_ms_cap  # noqa: E402
from review import report as review_report  # noqa: E402

import paths  # noqa: E402
import localgpu  # noqa: E402

HARNESS = Path(__file__).resolve().parent
STATE = paths.STATE                                  # run settings (written by autoresearch.py)
RAW = paths.RAW                                      # full job logs, never shown to the agent
OUTPUTS = paths.OUTPUTS                              # the agents' run folders: <date>/<tag>/
CAPS = json.loads((HARNESS / "caps.json").read_text())
SUBMITTED = []                                       # this call's GPU training tasks
COLS = ["name", "status", "val_loss", "step_ms", "train_step_ms", "total_params", "flops_ratio", "finished", "idea"]


def require_pool():
    """GPUs must be visible: agents' calls reach this through the launcher's relay, outside their jail"""
    if not localgpu.available():
        sys.exit("no GPUs are visible from here (this command runs through the experiment relay)")


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
    """a stopped ar-exp must not leave its trainings running, unrecorded"""
    for tid in SUBMITTED:
        rec = localgpu.get(tid)
        if not rec or rec["state"] not in localgpu.FINAL:
            localgpu.cancel(tid)
    print("ar-exp: stopped; its unfinished trainings were cancelled", flush=True)
    os._exit(128 + signum)


def wait(tid, poll=10):
    while True:
        rec = localgpu.get(tid)
        if rec and rec["state"] in localgpu.FINAL:
            return rec
        time.sleep(poll)


def check_report(c):
    """check.txt: the size / causality / compile report the agent sees"""
    if c.get("status") not in ("pass", "fail"):
        return "check: crash (it did not complete; see error.txt)\n"
    lines = [f"check: {c.get('status')}"]
    for comp, n in (c.get("components") or {}).items():
        cap = CAPS["component_caps"][comp]
        lines.append(f"  {comp:<9} {n:>12,} params  (cap {cap:,}; {n / max(CAPS['baseline_components'][comp], 1):.2f}x baseline)")
    if c.get("total_params"):
        lines.append(f"  total     {c['total_params']:>12,} params")
    if c.get("flops_ratio"):
        lines.append(f"  FLOPs per token: {c['flops_ratio']:.3f}x baseline (cap {CAPS['flops_ratio_cap']}x)")
    lines.append(f"  causality: {'FAILED' if c.get('causality_leaks') else 'ok'}")
    lines.append(f"  torch.compile: {'FAILED' if c.get('compile_error') else 'ok'}")
    for v in c.get("violations") or []:
        lines.append(f"VIOLATION: {v}")
    return "\n".join(lines) + "\n"


def run_one(run_dir, name, model_src):
    exp = run_dir / "experiments" / name
    raw = RAW / f"{run_dir.parent.name}__{run_dir.name}" / name
    exp.mkdir(parents=True)
    raw.mkdir(parents=True, exist_ok=True)
    model = exp / "model.py"
    shutil.copy(model_src, model)
    idea = Path(model_src).parent / "idea.md"
    if idea.exists():
        shutil.copy(idea, exp / "idea.md")
    out = dict(name=name, idea=first_line(exp / "idea.md"))
    meta = dict(submitted=time.time(), harness_commit=subprocess.run(
        ["git", "-C", str(HARNESS.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())

    # 1. the stored outcome of ar-check for this exact code (checked again on the snapshot)
    h = model_hash(model)
    rec = check_record(h)
    meta["check"] = dict(hash=h, when=rec and rec.get("when"), result=rec and rec.get("result"))
    if not rec or rec.get("result") != "pass":
        out.update(status="reject-check", reason="no passing ar-check for this exact model")
        return out, meta
    check = rec["check"]
    (exp / "check.txt").write_text(check_report(check) + (review_report(rec["review"]) if rec.get("review") else "")
                                   + f"  step time (8 GPUs, steps 20-120): {rec['step_ms']:.1f} ms/step (cap {step_ms_cap():.1f})\n")
    out.update(total_params=check.get("total_params"), flops_ratio=check.get("flops_ratio"), step_ms=rec["step_ms"])

    # 2. training on 8 GPUs (once more if it hung or lost its node)
    for attempt in (1, 2):
        tdir = raw / ("train" if attempt == 1 else "train_retry")
        tdir.mkdir(exist_ok=True)
        tid = localgpu.submit(f"bash {HARNESS / 'run_train.sh'} {model} {LR} {tdir}", gpus=paths.train_gpus(),
                              time="25m", project="autoresearch", workdir=str(tdir), log=str(tdir / "task.log"),
                              name=f"{run_dir.name}-{name}")
        SUBMITTED.append(tid)
        meta[f"train_task_{attempt}"] = tid
        rec = wait(tid)
        res = json.loads((tdir / "result.json").read_text()) if (tdir / "result.json").exists() else None
        infra = rec["state"] == "timeout" or (rec["state"] == "failed" and "allocation" in (rec.get("reason") or ""))
        if res or not infra or attempt == 2:
            break
    if (tdir / "train.log").exists():
        (exp / "train.log").write_text(train_curve((tdir / "train.log").read_text(errors="replace"), model))
    if res is None:
        err = model_error((tdir / "task.log").read_text(errors="replace") if (tdir / "task.log").exists() else "", model)
        if err:
            (exp / "error.txt").write_text(err)
            out.update(status="crash-train", reason="training crashed; see error.txt")
        else:
            out.update(status="crash-train", reason="training ended without a result (infrastructure; not your model)"
                       if rec["state"] in ("timeout", "failed") and not err else "training ended without a result")
    else:
        out.update(status="trained" if res["status"] == "ok" else res["status"], val_loss=res.get("val_loss"),
                   train_step_ms=res.get("check_step_ms"))
    return out, meta


def main(argv):
    require_pool()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    run_dir = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                                  text=True).stdout.strip() or ".").resolve()
    state = STATE / f"{run_dir.parent.name}__{run_dir.name}.json"
    if run_dir.parent.parent != OUTPUTS or not state.exists():
        fail("run this from your run folder")
    cfg = json.loads(state.read_text())
    if not argv or any("=" not in a for a in argv):
        fail("usage: ar-exp NAME=MODEL_FILE [NAME=MODEL_FILE ...]")
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
        rec = check_record(model_hash(src))
        if not rec:
            fail(f"{src} has not passed ar-check: run `ar-check {src}` first (ar-exp trains only models that "
                 "passed ar-check, with exactly the same code)")
        if rec.get("result") != "pass":
            fail(f"{src}: its ar-check result is FAIL ({rec.get('result')}, {rec.get('when')}); ar-exp trains only "
                 "models that passed ar-check")
        jobs.append((name, Path(src).resolve()))
    print(f"ar-exp: {len(jobs)} experiment(s), each passed ar-check: 2.0B-token training on 8 GPUs "
          f"(about 12-15 min); results follow", flush=True)
    with ThreadPoolExecutor(len(jobs)) as ex:
        pairs = list(ex.map(lambda j: run_one(run_dir, *j), jobs))
    results = [r for r, _ in pairs]

    tsv = run_dir / "results.tsv"
    if not tsv.exists():
        tsv.write_text("\t".join(COLS) + "\n")
    for r in results:
        r["finished"] = time.strftime("%Y-%m-%d %H:%M")
        (run_dir / "experiments" / r["name"] / "result.json").write_text(json.dumps(r, indent=2) + "\n")
        meta = next(m for rr, m in pairs if rr is r)
        (RAW / f"{run_dir.parent.name}__{run_dir.name}" / r["name"] / "meta.json").write_text(
            json.dumps({**r, **meta}, indent=2) + "\n")
        row = {**r, "val_loss": f"{r['val_loss']:.5f}" if r.get("val_loss") else "-",
               "step_ms": f"{r['step_ms']:.0f}" if r.get("step_ms") else "-",
               "train_step_ms": f"{r['train_step_ms']:.0f}" if r.get("train_step_ms") else "-",
               "flops_ratio": f"{r['flops_ratio']:.3f}" if r.get("flops_ratio") else "-"}
        with tsv.open("a") as f:
            f.write("\t".join(str(row.get(c) if row.get(c) is not None else "-") for c in COLS) + "\n")
        # one commit per experiment: its snapshot, checks, training logs, result row, and the run's state
        git(run_dir, "add", "-A", "--", ".", ":(exclude)trace")
        git(run_dir, "commit", "-q", "--no-verify", "-m",
            f"{r['name']}: {r['status']} {row['val_loss']}  {r['idea'][:100]}")
    print(f"\n{'name':24} {'status':18} {'val_loss':>9} {'ms/step':>8} {'(train)':>8}  idea / reason")
    for r in results:
        v = f"{r['val_loss']:.5f}" if r.get("val_loss") else "-"
        s = f"{r['step_ms']:.0f}" if r.get("step_ms") else "-"
        t = f"{r['train_step_ms']:.0f}" if r.get("train_step_ms") else "-"
        print(f"{r['name']:24} {r['status']:18} {v:>9} {s:>8} {t:>8}  {r.get('reason') or r['idea']}")
    print(f"\nbaseline: {cfg['baseline_val']:.5f} (lower is better). All results: results.tsv; details: experiments/<name>/")
    if cfg.get("budget"):
        used = len(tsv.read_text().splitlines()) - 2
        print(f"experiment budget: {used} of {cfg['budget']} used, {cfg['budget'] - used} left")


if __name__ == "__main__":
    main(sys.argv[1:])
