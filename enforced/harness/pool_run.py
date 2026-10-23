"""
pool_run.py -- FROZEN. Run a command on GPUs of the local GPU queue (common/localgpu.py) and wait for it:
prints the command's output (infrastructure details scrubbed) and exits with its exit code. Used by
check.py (ar-check) for the model check; agents have no command of their own that runs on GPUs.

usage: python harness/pool_run.py [--gpus N] [--time 15m] [--] COMMAND ...
       (runs in the current directory with the repository's env.sh sourced; N = 1..8 GPUs, default 1)

Tasks start in submission order as soon as enough GPUs are free.
"""

import os
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401  (puts common/ on sys.path)
import localgpu  # noqa: E402
from sanitize import model_error, scrub  # noqa: E402

HARNESS = Path(__file__).resolve().parent
ENV = f"source {HARNESS}/env.sh"
SUBMITTED = []          # GPU tasks submitted by this process (cancelled if it is stopped, see cancel_submitted)


def require_pool():
    """GPUs must be visible: agents' calls reach this through the launcher's relay, outside their jail"""
    if not localgpu.available():
        sys.exit("no GPUs are visible from here (this command runs through the experiment relay)")


def cancel_submitted():
    """cancel this process's unfinished GPU tasks (a stopped command must not leave work on the pool)"""
    for tid in SUBMITTED:
        rec = localgpu.get(tid)
        if not rec or rec["state"] not in localgpu.FINAL:
            localgpu.cancel(tid)


def run_on_pool(command, gpus=1, time_limit="15m", workdir=None, name=None, quiet=False, model_file=None):
    """submit `command` to the GPU queue, wait, print its output with infrastructure details removed; returns the
    exit code (124 on timeout). With model_file, an error shows only frames inside that file."""
    require_pool()
    tid = localgpu.submit(command, gpus=gpus, time=time_limit, project="autoresearch", workdir=workdir or os.getcwd(),
                          env=ENV + "; export TMPDIR=/tmp/$USER-$GQ_TASK_ID; mkdir -p $TMPDIR",
                          name=name, retries=1)
    SUBMITTED.append(tid)
    log = None
    if not quiet:
        print(f"waiting for {gpus} GPU(s) (limit {time_limit}) ...", flush=True)
    t0, announced = time.time(), False
    while True:
        rec = localgpu.get(tid) or {}
        if rec.get("state") == "running" and not announced and not quiet:
            print(f"running (waited {time.time() - t0:.0f} s)", flush=True)
            announced = True
        if rec.get("state") in localgpu.FINAL:
            break
        time.sleep(5)
    log = Path(rec["log"])
    text = log.read_text(errors="replace") if log.exists() else ""
    out = text
    if model_file and "Traceback (most recent call last)" in out:
        head = out[:out.index("Traceback (most recent call last)")]
        out = head + model_error(out, model_file)
    print(scrub(out.rstrip(), model_file), flush=True)
    if rec["state"] == "timeout":
        print(f"killed at its time limit ({time_limit})", flush=True)
        return 124
    if rec["state"] == "cancelled":
        return 130
    return rec.get("exit_code") if rec.get("exit_code") is not None else 1


def main(argv):
    gpus, limit = 1, "15m"
    while argv and argv[0] in ("--gpus", "--time"):
        if argv[0] == "--gpus":
            gpus = int(argv[1])
        else:
            limit = argv[1]
        argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        sys.exit(__doc__)
    command = argv[0] if len(argv) == 1 else " ".join(shlex.quote(a) for a in argv)
    return run_on_pool(command, gpus=gpus, time_limit=limit)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
