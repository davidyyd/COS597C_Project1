#!/usr/bin/env python3
"""
localgpu.py -- a minimal GPU task queue on the local machine, standing in for the cluster scheduler we used.

The harness submits every model check, speed benchmark and training here and polls for the result:

    import localgpu
    tid = localgpu.submit("bash run_train.sh model.py", gpus=8, time="25m", workdir=out_dir, log=out_dir + "/task.log")
    rec = localgpu.get(tid)     # rec["state"] in queued / running / done / failed / timeout / cancelled
    localgpu.cancel(tid)

Each submitted task gets a detached runner process (`python localgpu.py run TASK_ID`). Runners take GPUs strictly
in order of (priority, submission time), so a large task is never starved by small ones. A task sees only its
GPUs through CUDA_VISIBLE_DEVICES, is killed (process group) at its time limit or on cancel, and its record and
log stay in $AR_WORK/localgpu/. The number of GPUs is $AR_GPUS, or else what nvidia-smi lists.

    python common/localgpu.py status          # queued and running tasks, GPU leases
"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("AR_WORK", Path(__file__).resolve().parents[1] / "work")) / "localgpu"
TASKS, LEASES = ROOT / "tasks", ROOT / "leases"
QUEUED, RUNNING, DONE, FAILED, TIMEOUT, CANCELLED = "queued", "running", "done", "failed", "timeout", "cancelled"
FINAL = (DONE, FAILED, TIMEOUT, CANCELLED)
HOST = os.uname().nodename


def _now():
    return time.time()


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _seconds(limit):
    """'45m', '2h', '90s', '1:30:00' -> seconds"""
    s = str(limit).strip()
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    unit = {"s": 1, "m": 60, "h": 3600}.get(s[-1])
    return int(float(s[:-1]) * unit) if unit else int(s)


def num_gpus():
    if os.environ.get("AR_GPUS"):
        return int(os.environ["AR_GPUS"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except FileNotFoundError:
        return 0


def available():
    return num_gpus() > 0


def submit(command, gpus, time, project=None, workdir=None, env=None, log=None, name=None, retries=0, **_):
    """queue one task and start its runner; returns the task id"""
    for d in (TASKS, LEASES):
        d.mkdir(parents=True, exist_ok=True)
    n = num_gpus()
    if not 1 <= gpus <= n:
        raise ValueError(f"task needs {gpus} GPUs; this machine has {n}")
    tid = f"t{_now():.0f}-{uuid.uuid4().hex[:6]}"
    log = str(log or ROOT / "logs" / f"{tid}.log")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    rec = dict(id=tid, name=name, project=project, command=command, gpus=gpus, time_s=_seconds(time),
               workdir=str(workdir or os.getcwd()), env=env or "", log=log, priority=0, state=QUEUED,
               submitted=_now(), host=HOST)
    _write(TASKS / f"{tid}.json", rec)
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", tid], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return tid


def get(tid):
    return _read(TASKS / f"{tid}.json")


def cancel(tid):
    (TASKS / f"{tid}.cancel").touch()


def set_priority(tid, value):
    rec = get(tid)
    if rec and rec["state"] == QUEUED:
        rec["priority"] = int(value)
        _write(TASKS / f"{tid}.json", rec)


def _alive(pid):
    return pid and Path(f"/proc/{pid}").exists()


def _try_acquire(rec):
    """under the queue lock: GPUs for rec if it is first in line and enough are free, else None"""
    with open(ROOT / "queue.lock", "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        leases = {}
        for f in LEASES.glob("gpu*"):
            l = _read(f, {})
            if _alive(l.get("pid")):
                leases[int(f.name[3:])] = l
            else:
                f.unlink(missing_ok=True)
        queued = [r for r in (_read(f) for f in TASKS.glob("t*.json")) if r and r["state"] == QUEUED]
        first = min(queued, key=lambda r: (-r.get("priority", 0), r["submitted"]), default=None)
        if not first or first["id"] != rec["id"]:
            return None
        free = [g for g in range(num_gpus()) if g not in leases]
        if len(free) < rec["gpus"]:
            return None
        mine = free[:rec["gpus"]]
        for g in mine:
            _write(LEASES / f"gpu{g}", dict(task=rec["id"], pid=os.getpid()))
        rec.update(state=RUNNING, started=_now(), gpus_assigned=mine, runner_pid=os.getpid())
        _write(TASKS / f"{rec['id']}.json", rec)
        return mine


def _run(tid):
    path, flag = TASKS / f"{tid}.json", TASKS / f"{tid}.cancel"
    rec = _read(path)
    while (gpus := _try_acquire(rec)) is None:          # wait in line
        if flag.exists():
            rec.update(state=CANCELLED, ended=_now(), reason="cancelled while queued")
            _write(path, rec)
            return
        time.sleep(2)
        rec = _read(path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), GQ_TASK_ID=tid, GQ_ATTEMPT="1",
               GQ_JOB="local")
    script = f"{rec['env']}\ncd {rec['workdir']}\n{rec['command']}\n"
    with open(rec["log"], "a") as log:
        p = subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        state, reason = None, None
        while p.poll() is None:
            if flag.exists():
                state, reason = CANCELLED, "cancelled"
            elif _now() - rec["started"] > rec["time_s"]:
                state, reason = TIMEOUT, f"time limit {rec['time_s']} s"
            if state:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                break
            time.sleep(2)
    code = p.returncode
    rec.update(state=state or (DONE if code == 0 else FAILED), exit_code=code, ended=_now(),
               reason=reason or (None if code == 0 else f"exit code {code}"))
    _write(path, rec)
    for g in gpus:
        (LEASES / f"gpu{g}").unlink(missing_ok=True)


def status():
    for r in sorted((_read(f) for f in TASKS.glob("t*.json")), key=lambda r: r["submitted"]):
        if r and r["state"] not in FINAL:
            print(f"{r['id']}  {r['state']:8} {r['gpus']} GPU(s)  {r.get('name') or ''}  {r['command'][:80]}")
    print(f"GPUs: {num_gpus()}; leased: {sorted(int(f.name[3:]) for f in LEASES.glob('gpu*'))}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        _run(sys.argv[2])
    elif sys.argv[1:2] == ["status"]:
        status()
    else:
        sys.exit(__doc__)
