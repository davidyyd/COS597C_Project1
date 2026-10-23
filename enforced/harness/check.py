"""
check.py -- FROZEN. The gate every model passes before it is trained (installed as the `ar-check`
command). ar-exp trains only a model whose exact code (see model_hash) has passed ar-check: it
looks the result up here and never checks or benchmarks by itself.

  ar-check MODEL_FILE

  1. model check on one GPU (smoke.py): the model file's source rules, parameters per
     component vs the caps, FLOPs per token, causality, torch.compile, the optimizer of every
     parameter;
  (runs started with --review) complexity review (review.py): a separate Claude call scores the whole
     change relative to the baseline; it runs alongside the model check;
  2. speed benchmark on 8 GPUs (train.py --mode speed): the real training steps, with the
     step time measured over steps 20-120 and checked against the step-time cap. Training itself
     only records its step time; this measurement is the one that counts.
Prints the check report, then the measured step time, and a final line CHECK: PASS or
CHECK: FAIL (...). Exit code 0 on PASS.

Stopping ar-check (Ctrl-C, a timeout, the agent's session ending) cancels its GPU tasks.

Each model is benchmarked once: the measurement is stored under a hash of the model's code
(its syntax tree, so comments and formatting do not count) and reused for the same model.
Resubmitting an unchanged model does not draw a new measurement. The outcome of every ar-check
(PASS or FAIL, the check report, the step time) is stored under the same hash in $AR_WORK/state/enforced/check_cache/,
which is where ar-exp looks; outcomes stored under older check rules (CHECK_RULES) do not count.
"""

import ast
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import localgpu  # noqa: E402
from pool_run import SUBMITTED, cancel_submitted, require_pool, run_on_pool  # noqa: E402
from prepare import LR  # noqa: E402
import review  # noqa: E402

HARNESS = Path(__file__).resolve().parent
SNAPSHOTS = paths.SNAPSHOTS
SPEED_CACHE = paths.STATE / "speed_cache"      # {model hash}.json, outside the agents' reach
CHECK_CACHE = paths.STATE / "check_cache"      # {model hash}.json: the last ar-check outcome
# version of the check's rules: an outcome stored under older rules counts as no outcome (the model must
# be checked again). Bump it whenever smoke.py, static_check.py or precision_check.py get stricter.
CHECK_RULES = "2026-10-12-precision"


def rules():
    """the rules this check runs under: CHECK_RULES, plus the review's version in runs with the review"""
    return CHECK_RULES + (f"+{review.VERSION}" if review.enabled() else "")


def step_ms_cap():
    caps = json.loads((HARNESS / "caps.json").read_text())
    return caps.get("step_ms_cap") or caps["step_time_ratio_cap"] * caps["baseline_step_ms"]


def model_hash(model_file):
    """hash of the model's code, ignoring comments, formatting and whitespace"""
    src = Path(model_file).read_text()
    try:
        key = ast.dump(ast.parse(src), annotate_fields=False, include_attributes=False)
    except SyntaxError:
        key = src
    return hashlib.sha256(key.encode()).hexdigest()


def check_record(h):
    """the stored outcome of the last ar-check of the model with hash h, or None"""
    f = CHECK_CACHE / f"{h}.json"
    rec = json.loads(f.read_text()) if f.exists() else None
    return rec if rec and rec.get("rules") == rules() else None


def store(h, **fields):
    CHECK_CACHE.mkdir(parents=True, exist_ok=True)
    (CHECK_CACHE / f"{h}.json").write_text(json.dumps(dict(hash=h, when=time.strftime("%Y-%m-%d %H:%M"),
                                                           rules=rules(), **fields),
                                                      indent=2) + "\n")


def speed_test(model_file, out_dir, name, priority=None):
    """8-GPU speed benchmark of model_file (a frozen snapshot). Returns (ms per step or None, task records).
    A model measured before (same code) returns its stored measurement (task records then hold one
    {"cached": ...} entry). A benchmark that times out or loses its node is run once more."""
    SPEED_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SPEED_CACHE / f"{model_hash(model_file)}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        return c["ms"], [{"cached": c}]
    recs = []
    for attempt in (1, 2):
        d = Path(out_dir) / ("speed" if attempt == 1 else "speed_retry")
        d.mkdir(parents=True, exist_ok=True)
        tid = localgpu.submit(f"bash {HARNESS / 'run_train.sh'} {model_file} {LR} {d} speed", gpus=paths.train_gpus(), time="15m",
                              project="autoresearch", workdir=str(d), log=str(d / "task.log"), name=name)
        SUBMITTED.append(tid)
        if priority is not None:
            localgpu.set_priority(tid, priority)
        while True:
            rec = localgpu.get(tid)
            if rec and rec["state"] in localgpu.FINAL:
                break
            time.sleep(10)
        recs.append(rec)
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
        if res and res.get("status") == "speed":
            cache.write_text(json.dumps(dict(ms=res["check_step_ms"], host=rec.get("host"), task=rec.get("id"),
                                             model=str(model_file), when=time.strftime("%Y-%m-%d %H:%M")), indent=2))
            return res["check_step_ms"], recs
        infra = rec["state"] == "timeout" or (rec["state"] == "failed" and "allocation" in (rec.get("reason") or ""))
        if not infra:
            break
    return None, recs


def stop(signum, _frame):
    cancel_submitted()
    print("ar-check: stopped; its unfinished GPU tasks were cancelled", flush=True)
    os._exit(128 + signum)


def main(argv):
    require_pool()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    if len(argv) != 1 or not Path(argv[0]).is_file():
        sys.exit("usage: ar-check MODEL_FILE")
    src = Path(argv[0]).resolve()
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    snap_dir = SNAPSHOTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{src.parent.name}"
    snap_dir.mkdir()
    snap = snap_dir / src.name
    shutil.copy(src, snap)
    h = model_hash(snap)
    reviewed = review.enabled()
    if reviewed:
        idea = src.parent / "idea.md"
        if idea.exists():
            shutil.copy(idea, snap_dir / "idea.md")
        import threading
        rev = {}
        def _review():
            try:
                rev["v"] = review.review(snap, snap_dir / "idea.md")
            except Exception as e:
                rev["error"] = str(e)
        rev_thread = threading.Thread(target=_review, daemon=True)
        rev_thread.start()
    print(f"checking {src.name}: model check on one GPU"
          + (" and complexity review" if reviewed else "") + ", then a speed benchmark on 8 GPUs", flush=True)
    rc = run_on_pool(f"python {HARNESS / 'smoke.py'} {snap} --json {snap_dir / 'check.json'}", gpus=1,
                     time_limit="20m", workdir=str(src.parent), name="check", model_file=snap)
    check = json.loads((snap_dir / "check.json").read_text()) if (snap_dir / "check.json").exists() else {"status": "crash"}
    if rc != 0:
        store(h, result="fail (model check)", check=check, step_ms=None, snapshot=str(snap))
        print("CHECK: FAIL (model check)")
        return 1
    verdict = None
    if reviewed:
        rev_thread.join()
        if "error" in rev:           # not the model's fault: not stored, so checking again retries
            print(f"\n{rev['error']}")
            print("CHECK: FAIL (complexity review unavailable; run ar-check again)")
            return 1
        verdict = rev["v"]
        print("\n" + review.report(verdict), end="", flush=True)
        if not verdict["passed"]:
            store(h, result="fail (complexity review)", check=check, review=verdict, step_ms=None, snapshot=str(snap))
            print("CHECK: FAIL (complexity review)")
            return 1
    cap = step_ms_cap()
    print(f"\nspeed benchmark: training steps on 8 GPUs, timed over steps 20-120 (cap {cap:.1f} ms/step) ...",
          flush=True)
    ms, recs = speed_test(snap, snap_dir, "check-speed")
    if recs and "cached" in recs[0]:
        print(f"(this model was benchmarked before, {recs[0]['cached']['when']}: each model is measured once)")
    if ms is None:
        out = (snap_dir / "speed" / "task.log")
        from sanitize import model_error
        err = model_error(out.read_text(errors="replace"), snap) if out.exists() else ""
        print(err or "the speed benchmark ended without a result")
        store(h, result="fail (speed benchmark crashed)", check=check, review=verdict, step_ms=None, snapshot=str(snap))
        print("CHECK: FAIL (speed benchmark crashed)")
        return 1
    print(f"step time: {ms:.1f} ms/step (cap {cap:.1f})")
    if ms > cap:
        store(h, result="fail (step time)", check=check, review=verdict, step_ms=ms, snapshot=str(snap))
        print("CHECK: FAIL (step time)")
        return 1
    store(h, result="pass", check=check, review=verdict, step_ms=ms, snapshot=str(snap))
    print("CHECK: PASS (ar-exp can now train this model)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
