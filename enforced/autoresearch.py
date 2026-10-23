#!/usr/bin/env python3
"""
autoresearch.py -- operator tool: start, resume, stop and list Karpathy-style autoresearch runs (enforced arm).

  python autoresearch.py start --strategy search --batch 4 [--budget 400]   # Claude Code, jailed
  python autoresearch.py start --strategy search --budget 8 --tag try --no-jail   # a short unjailed test
  python autoresearch.py start --run 2026-10-12/search-claude-medium        # resume a run
  python autoresearch.py stop  --run 2026-10-12/search-claude-medium        # ends it when its session next stops
  python autoresearch.py status
  python autoresearch.py instructions --strategy search                   # print the agent's CLAUDE.md
  python autoresearch.py trace --run 2026-10-12/search-claude-medium [--segment N]   # readable trace
(launch.sh starts runs in the background; tools/audit.py audits a finished run.)

A run is ONE Claude Code session that runs the research loop itself, in its own folder
$AR_WORK/outputs/<start date>/<tag>/ (a small git repository of its own, whose first commit is the base:
the baseline files of the repository commit the run was started from). The folder holds only what the
agent needs:
  CLAUDE.md      mission, setup, baseline score, model interface, constraints, this run's loop
                 (programs/<strategy>.md), commands (ar-check, ar-exp), rules
  model.py       the model it edits (starts as the baseline); baseline_model.py (read-only)
  notes.md       its notes; results.tsv and experiments/ (written by ar-exp)
Everything else (harness, this launcher, templates, run settings, data, the GPUs) stays outside the
run folder and is never copied into it.

The agent runs in a jail (tools/jail.py, needs bubblewrap) that shows only its run folder; ar-check and
ar-exp reach the harness through a relay (the agent's stub writes a request into RUN/.ar/, this launcher
runs the real command). Without bubblewrap, or with --no-jail, the agent runs unjailed (see tools/jail.py).
A run starts only if the repository has no uncommitted changes (or --allow-dirty) and
baseline/baseline.json was measured on harness/baseline_model.py; the run records the commit.

Login: the agent and the auditor use the token in ~/.ar_claude_token (from `claude setup-token`; mode 600),
or else ANTHROPIC_API_KEY. A jailed agent needs one of the two (its home is empty); an unjailed agent
without either uses your own `claude` login.

The launcher keeps the record of a run in $AR_WORK/traces/<date>/<tag>/, outside the run folder
and the jail: trace.jsonl (the full session stream, all segments, each after a {"type": "segment"}
marker line), requests.jsonl (every relayed command: arguments, times, exit code), background/
(outputs of commands the agent ran in the background) and run.json (settings, base commit,
per-segment hashes). A session is resumed whenever it stops (the agent ended its turn, or Claude
exited), and killed and resumed only if it has been silent for IDLE_TIMEOUT with nothing running.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import jail  # noqa: E402
import paths  # noqa: E402  (harness/paths.py, on sys.path through jail)

MASTER = Path(__file__).resolve().parent
OUTPUTS, TRACES, STATE = paths.OUTPUTS, paths.TRACES, paths.STATE
CONTROL = STATE / "control"
SETTINGS = MASTER / "harness" / "agent_settings.json"
IDLE_TIMEOUT = 2 * 3600               # a session silent this long, with no relayed command running, is restarted
RELAYED = ("ar-exp", "ar-check")
AUTH_ERROR = re.compile(r"OAuth|authenticat|unauthori[sz]ed|invalid (api key|token)|\b401\b", re.I)
DEFAULT_MODEL, DEFAULT_EFFORT = "claude-opus-5-5", "medium"


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def key(run):                          # "2026-10-02/linear-claude-medium" -> state file name
    return run.replace("/", "__")


def git(wt, *args):
    return subprocess.run(["git", "-C", str(wt), "-c", "user.name=autoresearch", "-c",
                           "user.email=autoresearch@local", *args], capture_output=True, text=True, check=True).stdout


def prompts(cfg):
    f = "CLAUDE.md"
    return (f"Read {f}, then start your research loop and keep it going.",
            f"Continue your research loop where you left off ({f} has your instructions; notes.md, results.tsv "
            "and experiments/ have the full state). Keep the loop going, and never end your turn while an "
            "ar-check or ar-exp is running: ending your turn ends your session and discards their results.")


# ---------------------------------------------------------------- traces

def block(t, src, n=20000):
    t = str(t)
    t = t if len(t) <= n else t[:n] + f"\n... [{len(t) - n} more characters in {src}]"
    return "```\n" + t.replace("```", "ˋˋˋ") + "\n```"


def render(lines, name, prompt):
    """readable markdown of one session segment (claude -p stream-json)"""
    stream = Path(name)
    parts = [f"# Session {stream.stem}\n", "## Prompt\n\n" + prompt]
    k = 0
    for raw in lines:
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = e.get("type")
        if t == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "text" and c["text"].strip():
                    parts.append("## Agent\n\n" + c["text"].strip())
                elif c.get("type") == "thinking" and c.get("thinking", "").strip():
                    parts.append("*Thinking:* " + c["thinking"].strip())
                elif c.get("type") == "tool_use":
                    k += 1
                    inp = c.get("input") or {}
                    parts.append(f"### Tool call {k}: {c.get('name')}\n\n" + block(inp.get("command") or json.dumps(inp, indent=2), stream.name))
        elif t == "user":
            for c in e["message"].get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    content = c.get("content")
                    if isinstance(content, list):
                        content = "\n".join(x.get("text", "[image]") for x in content if isinstance(x, dict))
                    parts.append(("**Result (error):**" if c.get("is_error") else "**Result:**") + "\n\n" + block(content, stream.name))
        elif t == "system" and e.get("subtype") == "compact_boundary":
            parts.append("---\n\n*Conversation compacted here (older messages summarized).*")
        elif t == "result":
            parts.append(f"---\n\n*Segment ended: {e.get('subtype')}, {e.get('num_turns')} model calls, "
                         f"{(e.get('duration_ms') or 0) / 60000:.0f} min*")
    return "\n\n".join(parts) + "\n"


def show_trace(run_id, segment=None):
    """print the readable view of a run's trace (all segments, or one)"""
    seg = None
    for raw in (TRACES / run_id / "trace.jsonl").read_text().splitlines():
        if raw.startswith('{"type": "segment"'):
            if seg and segment in (None, seg[0]):
                print(render(seg[2], f"segment_{seg[0]:03d}", seg[1]))
            m = json.loads(raw)
            seg = (m["n"], m.get("prompt", ""), [])
        elif seg:
            seg[2].append(raw)
    if seg and segment in (None, seg[0]):
        print(render(seg[2], f"segment_{seg[0]:03d}", seg[1]))


# ---------------------------------------------------------------- runs

REVIEW_TEXT = """
4. **Method complexity** (`ar-check` in this run): a separate reviewer reads the whole change of your
   model relative to `baseline_model.py` (always the baseline, not the experiment it builds on) and
   scores how complex it is as a method, from 1 (one formula-level change) to 10 (a pile of changes
   with no common idea). It sees only the code (without comments) and your `idea.md`, never results.
   A model passes only with complexity ≤ {MAX}: about one mechanism with the parts it needs. Several
   mechanisms with separate purposes, hand-set constants, hand-picked layers or special cases raise
   the score. Building on earlier results is fine as long as the whole method stays this simple: to
   add an idea to a model near the limit, simplify or remove something else. The review is part of
   the stored check result: checking the same code again returns the same review.
"""


def render_instructions(strategy, batch, budget, review=False):
    """the agent's CLAUDE.md: template/instructions.md with programs/<strategy>.md as its loop,
    and the baseline score and caps filled in from baseline/baseline.json and harness/caps.json"""
    base = json.loads((MASTER / "baseline" / "baseline.json").read_text())
    caps = json.loads((MASTER / "harness" / "caps.json").read_text())
    loop = (MASTER / "programs" / f"{strategy}.md").read_text().replace("{BATCH}", str(batch)).strip()
    table = ["   | component | baseline | cap |", "   |---|---|---|"]
    for comp, n in caps["baseline_components"].items():
        label = "other (norms, gates, scalars)" if comp == "other" else comp
        flat = " (flat allowance)" if comp == "other" else ""
        table.append(f"   | {label} | {n:,} | {caps['component_caps'][comp]:,}{flat} |")
    ratio = caps["component_caps"]["attention"] / caps["baseline_components"]["attention"]
    lr = re.search(r'^LR = "([^"]+)"', (MASTER / "harness" / "prepare.py").read_text(), re.M).group(1)
    text = ((MASTER / "template" / "instructions.md").read_text()
            .replace("{LOOP}", loop).replace("{BATCH}", str(batch)).replace("{BUDGET}", str(budget))
            .replace("{BASELINE_VAL}", f"{base['best_val']:.5f}").replace("{LR}", lr)
            .replace("{STEP_CAP}", f"{caps['step_ms_cap']:.1f}").replace("{STEP_BASE}", f"{caps['baseline_step_ms']:.1f}")
            .replace("{FLOPS_CAP}", f"{caps['flops_ratio_cap']}").replace("{COMPONENT_CAP}", f"{ratio:.2f}")
            .replace("{CAPS_TABLE}", "\n".join(table))
            .replace("{REVIEW}", REVIEW_TEXT.replace("{MAX}", str(_review_max())) if review else ""))
    return text


def _review_max():
    sys.path.insert(0, str(MASTER / "harness"))
    import review
    return review.MAX_COMPLEXITY


def create(args):
    run = f"{datetime.now().strftime('%Y-%m-%d')}/{args.tag or f'{args.strategy}-claude-{args.effort}'}"
    wt = OUTPUTS / run
    if wt.exists() or (STATE / f"{key(run)}.json").exists():
        sys.exit(f"run {run} already exists; pick another --tag (or resume it with --run {run})")
    dirty = git(paths.REPO, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty and not args.allow_dirty:
        sys.exit("the repository has uncommitted changes; commit them first (a run records the commit it starts "
                 f"from), or pass --allow-dirty:\n{dirty}")
    base = json.loads((MASTER / "baseline" / "baseline.json").read_text())
    caps = json.loads((MASTER / "harness" / "caps.json").read_text())
    model_sha = hashlib.sha256((MASTER / "harness" / "baseline_model.py").read_bytes()).hexdigest()
    if base.get("model_sha256") != model_sha:
        sys.exit("baseline/baseline.json was not measured on harness/baseline_model.py (model_sha256 differs): "
                 "measure the baseline of this model and commit it first")
    commit = git(paths.REPO, "rev-parse", "HEAD").strip() + ("-dirty" if dirty else "")
    batch = 1 if args.strategy == "linear" else args.batch
    instructions = render_instructions(args.strategy, batch, args.budget, review=args.review)
    wt.mkdir(parents=True)
    (wt / "CLAUDE.md").write_text(instructions)
    shutil.copy(MASTER / "harness" / "baseline_model.py", wt / "baseline_model.py")
    shutil.copy(MASTER / "harness" / "baseline_model.py", wt / "model.py")
    (wt / "notes.md").write_text("# Notes\n\n(nothing yet)\n")
    (wt / "results.tsv").write_text("name\tstatus\tval_loss\tstep_ms\ttrain_step_ms\ttotal_params\tflops_ratio\t"
                                    "finished\tidea\n"
                                    f"baseline\tbaseline\t{base['best_val']:.5f}\t{caps['baseline_step_ms']:.0f}\t"
                                    f"{caps['baseline_step_ms']:.0f}\t{caps['total_params']}\t1.000\t-\t"
                                    "baseline model (baseline_model.py)\n")
    (wt / ".gitignore").write_text("cands/\n__pycache__/\n.ar/\n")
    (wt / "experiments").mkdir()
    git(wt, "init", "-q", "-b", "main")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", f"start: {args.strategy}, batch {batch}, claude {args.model} "
                                  f"effort={args.effort}; base {commit[:7]}")
    cfg = dict(run=run, strategy=args.strategy, batch=batch, budget=args.budget, agent="claude", model=args.model,
               effort=args.effort, review=bool(args.review), jail=not args.no_jail,
               baseline_val=base["best_val"], session_id=str(uuid.uuid4()), segments=0,
               base_commit=commit, base_subject=git(paths.REPO, "log", "-1", "--format=%s").strip(),
               started=datetime.now().isoformat(timespec="seconds"))
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{key(run)}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (TRACES / run).mkdir(parents=True)
    write_run_json(cfg)
    log(f"created {wt}")
    return cfg


def write_run_json(cfg, segments=None):
    """$AR_WORK/traces/<run>/run.json: what the run is (settings, base commit) and per-segment trace hashes"""
    f = TRACES / cfg["run"] / "run.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    claude = subprocess.run([str(jail.claude_binary()), "--version"], capture_output=True, text=True).stdout.strip()
    meta = dict(run=cfg["run"], state=cfg, claude_version=old.get("claude_version") or claude,
                jail=dict(enabled=jailed(cfg), limits=jail.LIMITS if jail.systemd_scope() else None,
                          read_only=jail.READ_ONLY_IN_RUN),
                settings=json.loads(SETTINGS.read_text()), segments=segments if segments is not None else
                old.get("segments", []))
    f.write_text(json.dumps(meta, indent=2) + "\n")


def jailed(cfg):
    """whether this run's agent runs in the jail: asked for at start (not --no-jail) and bubblewrap is here"""
    return cfg.get("jail", True) and jail.available()


def agent_cmd(cfg, first, prompt):
    # inside the jail the binary is mounted at /ar/claude; the settings are passed inline (the settings
    # file itself is outside the jail)
    return ["/ar/claude" if jailed(cfg) else str(jail.claude_binary()), "-p", prompt, "--model", cfg["model"],
            "--effort", cfg["effort"], "--output-format", "stream-json", "--verbose", "--setting-sources", "project",
            "--settings", json.dumps(json.loads(SETTINGS.read_text())), "--strict-mcp-config",
            *(["--session-id", cfg["session_id"]] if first else ["--resume", cfg["session_id"]])]


def experiments_used(wt):
    """experiments recorded in the run's results.tsv (the baseline row excluded)"""
    f = Path(wt) / "results.tsv"
    return max(0, len(f.read_text().splitlines()) - 2) if f.exists() else 0


def experiments_running(wt):
    """an ar-exp call (harness/run_experiment.py) is running in this run folder"""
    out = subprocess.run(["pgrep", "-f", "harness/run_experiment.py"], capture_output=True, text=True).stdout
    for pid in out.split():
        try:
            if Path(os.readlink(f"/proc/{pid}/cwd")).resolve() == Path(wt).resolve():
                return True
        except OSError:
            pass
    return False


class Relay(threading.Thread):
    """runs the jailed agent's ar-check / ar-exp calls outside the jail. The agent's stub writes
    RUN/.ar/<id>.req.json; this thread validates it, runs tools/bin/<tool> in the requested folder (inside
    the run folder only), streams its output to RUN/.ar/<id>.out, writes the exit code to RUN/.ar/<id>.rc,
    kills it if RUN/.ar/<id>.cancel appears, and logs every call to the run's requests.jsonl."""

    def __init__(self, cfg, env):
        super().__init__(daemon=True)
        self.wt, self.env = OUTPUTS / cfg["run"], env
        self.q, self.log = self.wt / ".ar", TRACES / cfg["run"] / "requests.jsonl"
        self.procs, self.done = {}, threading.Event()

    def record(self, **ev):
        with self.log.open("a") as f:
            f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), **ev)) + "\n")

    def finish(self, rid, code, out_text=None):
        if out_text is not None:
            (self.q / f"{rid}.out").write_text(out_text)
        tmp = self.q / f".{rid}.rc"
        tmp.write_text(str(code))
        os.replace(tmp, self.q / f"{rid}.rc")

    def start_request(self, f):
        rid = f.name[:-len(".req.json")]
        try:
            r = json.loads(f.read_text())
            os.replace(f, self.q / f"{rid}.taken")
            tool, argv, cwd = r["tool"], [str(x) for x in r["argv"]], Path(r["cwd"]).resolve()
        except (OSError, ValueError, KeyError, TypeError):
            return
        why = None
        if tool not in RELAYED:
            why = f"unknown command {tool}"
        elif cwd != self.wt.resolve() and self.wt.resolve() not in cwd.parents:
            why = "commands run only inside the run folder"
        if why:
            self.record(id=rid, tool=tool, argv=argv, cwd=str(cwd), refused=why)
            self.finish(rid, 2, f"{tool}: {why}\n")
            return
        out = open(self.q / f"{rid}.out", "wb")
        p = subprocess.Popen([str(MASTER / "tools" / "bin" / tool), *argv], cwd=cwd, env=self.env, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.procs[rid] = (p, out, time.time(), tool)
        self.record(id=rid, tool=tool, argv=argv, cwd=str(cwd), pid=p.pid, event="start")

    def run(self):
        while not self.done.is_set():
            for f in sorted(self.q.glob("*.req.json")):
                self.start_request(f)
            for rid, (p, out, t0, tool) in list(self.procs.items()):
                if (self.q / f"{rid}.cancel").exists() and p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    (self.q / f"{rid}.cancel").unlink()
                    self.record(id=rid, event="cancelled by the agent")
                if p.poll() is not None:
                    out.close()
                    self.finish(rid, p.returncode)
                    self.record(id=rid, tool=tool, event="end", exit=p.returncode, minutes=round((time.time() - t0) / 60, 1))
                    del self.procs[rid]
            time.sleep(1)

    def busy(self):
        return bool(self.procs)


def run_segment(cfg, k, prompt, env, relay):
    """one session segment: start (or resume) the agent, append its stream to trace.jsonl; returns how it ended"""
    run_id, wt = cfg["run"], OUTPUTS / cfg["run"]
    trace = TRACES / run_id / "trace.jsonl"
    first = k == 1
    argv = agent_cmd(cfg, first, prompt)
    if jailed(cfg):
        argv = jail.bwrap_argv(run_id, env) + argv
    with trace.open("a") as ft:
        ft.write(json.dumps({"type": "segment", "n": k, "started": datetime.now().isoformat(timespec="seconds"),
                             "prompt": prompt}) + "\n")
        ft.flush()
        # jailed: the host environment for the process started here (systemd-run needs the user session's
        # variables to reach the user systemd); the agent gets only `env`, through bwrap's --clearenv
        p = subprocess.Popen(argv, cwd=wt, env=relay.env if jailed(cfg) else env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        last = [time.time()]
        err = []
        threading.Thread(target=lambda: err.extend(p.stderr), daemon=True).start()

        secrets = [v for k, v in (env or {}).items() if k in jail.SECRET_VARS and v]

        def copy():
            for line in p.stdout:
                line = line.decode(errors="replace")
                for v in secrets:                    # a login the agent printed never reaches the trace
                    line = line.replace(v, "<redacted>")
                ft.write(line)
                ft.flush()
                last[0] = time.time()
        reader = threading.Thread(target=copy, daemon=True)
        reader.start()
        how = None
        while how is None:
            try:
                how = f"exit {p.wait(timeout=30)}"
            except subprocess.TimeoutExpired:
                idle = time.time() - last[0]
                if idle > IDLE_TIMEOUT and not relay.busy() and not experiments_running(wt):
                    p.kill()
                    p.wait()
                    how = f"killed after {idle / 60:.0f} min without output"
        reader.join(timeout=60)
        if err:
            ft.write(json.dumps({"type": "stderr", "n": k, "text": b"".join(err).decode(errors="replace")}) + "\n")
    return how


def run(args, cfg):
    run_id = cfg["run"]
    wt, sf = OUTPUTS / run_id, STATE / f"{key(run_id)}.json"
    stop = CONTROL / f"{key(run_id)}.stop"
    CONTROL.mkdir(parents=True, exist_ok=True)
    (TRACES / run_id).mkdir(parents=True, exist_ok=True)
    (wt / ".ar").mkdir(exist_ok=True)
    # the relay runs outside the jail with the launcher's environment
    host_env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
    host_env.update(AR_RUN=run_id, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
                    PATH=f"{MASTER / 'tools' / 'bin'}:{MASTER / 'tools' / 'agent_bin'}:{jail.PYBIN}:{os.environ['PATH']}")
    if getattr(args, "no_jail", False):
        cfg["jail"] = False
    auth = paths.claude_auth()
    if jailed(cfg):
        if not auth:
            sys.exit(f"no agent login: run `claude setup-token` and save the token in {paths.TOKEN_FILE} (chmod 600), "
                     "or set ANTHROPIC_API_KEY")
        env = jail.base_env(run_id, CUDA_VISIBLE_DEVICES="", **auth)
        if not jail.probe(run_id, quiet=True):
            sys.exit(f"{run_id}: the jail leak check failed (see above); not starting")
    else:
        log(f"{run_id}: WARNING: the agent runs WITHOUT the jail "
            f"({'--no-jail' if cfg.get('jail') is False else 'bubblewrap (bwrap) is not installed'}): it can read "
            "the harness and everything else you can; ar-check and ar-exp still go through the relay")
        env = jail.unjailed_env(run_id, **(auth or {}))
    relay = Relay(cfg, host_env)
    relay.start()
    kickoff, cont = prompts(cfg)
    quick_failures = 0
    while not stop.exists():
        if cfg.get("budget") and experiments_used(wt) >= cfg["budget"] and not relay.busy():
            log(f"{run_id}: experiment budget used up ({experiments_used(wt)} of {cfg['budget']}); the run ends")
            break
        k = cfg["segments"] + 1
        log(f"{run_id}: segment {k} ({'start' if k == 1 else 'resume'})")
        t0 = time.time()
        how = run_segment(cfg, k, kickoff if k == 1 else cont, env, relay)
        minutes = (time.time() - t0) / 60
        trace = TRACES / run_id / "trace.jsonl"
        cfg["segments"] = k
        sf.write_text(json.dumps(cfg, indent=2) + "\n")
        segs = json.loads((TRACES / run_id / "run.json").read_text()).get("segments", [])
        segs.append(dict(n=k, how=how, minutes=round(minutes), ended=datetime.now().isoformat(timespec="seconds"),
                         trace_bytes=trace.stat().st_size, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
        write_run_json(cfg, segs)
        if jailed(cfg):                    # outputs of commands the agent ran in the background
            try:                           # files can vanish while being copied; never stop the run for that
                shutil.copytree(jail.agent_dirs(run_id)[1], TRACES / run_id / "background", dirs_exist_ok=True)
            except (shutil.Error, OSError) as e:
                log(f"{run_id}: some background outputs were not copied ({str(e)[:300]})")
        git(wt, "commit", "-q", "--allow-empty", "-m", f"session segment {k} ended ({how}, {minutes:.0f} min)")
        log(f"{run_id}: segment {k} ended ({how}, {minutes:.0f} min)")
        with trace.open("rb") as f:                # a login problem: keep retrying, never give up on it
            f.seek(max(0, trace.stat().st_size - 20000))
            tail = f.read().decode(errors="replace")
        if minutes < 2 and AUTH_ERROR.search(tail):
            log(f"{run_id}: segment {k} failed to log in; retrying in 10 min (check {paths.TOKEN_FILE})")
            time.sleep(600)
            continue
        quick_failures = quick_failures + 1 if minutes < 2 else 0
        if quick_failures >= 5:
            log(f"{run_id}: 5 segments in a row ended within 2 minutes; stopping (see {trace})")
            break
        time.sleep(min(600, 15 * 2 ** quick_failures) if quick_failures else 5)
    relay.done.set()
    if stop.exists():
        stop.unlink()
        log(f"{run_id}: stopped by the operator")


def main():
    os.umask(0o077)                    # run folders, traces, state and raw logs: readable by the owner only
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="start a new run (--strategy ...) or resume one (--run DATE/TAG)")
    s.add_argument("--run", help="resume this run (DATE/TAG)")
    s.add_argument("--strategy", choices=["linear", "batched", "search"])
    s.add_argument("--batch", type=int, default=4, help="experiments per round (batched, search)")
    s.add_argument("--budget", type=int, default=400, help="experiments (ar-exp) per run; the run ends when used up")
    s.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    s.add_argument("--effort", default=DEFAULT_EFFORT, help=f"reasoning effort (default: {DEFAULT_EFFORT})")
    s.add_argument("--tag", help="run name under today's date (default: <strategy>-claude-<effort>)")
    s.add_argument("--review", action="store_true", help="ar-check also runs the complexity review (harness/review.py)")
    s.add_argument("--no-jail", action="store_true", help="run the agent without the bubblewrap jail (for testing)")
    s.add_argument("--allow-dirty", action="store_true", help="start even if the repository has uncommitted changes")
    s = sub.add_parser("stop")
    s.add_argument("--run", required=True)
    sub.add_parser("status")
    s = sub.add_parser("instructions", help="print the instructions an agent would get (no run is created)")
    s.add_argument("--strategy", choices=["linear", "batched", "search"], default="search")
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--budget", type=int, default=400)
    s.add_argument("--review", action="store_true")
    s = sub.add_parser("trace", help="print the readable view of a run's trace")
    s.add_argument("--run", required=True)
    s.add_argument("--segment", type=int)
    a = ap.parse_args()
    if a.cmd == "start":
        if a.run:
            run(a, json.loads((STATE / f"{key(a.run)}.json").read_text()))
        elif a.strategy:
            run(a, create(a))
        else:
            sys.exit("give --strategy for a new run, or --run DATE/TAG to resume one")
    elif a.cmd == "instructions":
        print(render_instructions(a.strategy, 1 if a.strategy == "linear" else a.batch, a.budget, review=a.review))
    elif a.cmd == "trace":
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # allow `| head`
        show_trace(a.run, a.segment)
    elif a.cmd == "stop":
        CONTROL.mkdir(parents=True, exist_ok=True)
        (CONTROL / f"{key(a.run)}.stop").write_text("")
        print(f"{a.run} stops when its session next ends (the session itself is not interrupted)")
    else:
        for sf in sorted(STATE.glob("*.json")):
            cfg = json.loads(sf.read_text())
            rows = (OUTPUTS / cfg["run"] / "results.tsv").read_text().splitlines()[1:]
            vals = [float(r.split("\t")[2]) for r in rows if r.split("\t")[2] not in ("-", "")]
            print(f"{cfg['run']:40} {cfg['strategy']:8} batch {cfg['batch']}  {cfg['model']} {cfg['effort']:6} "
                  f"segments {cfg['segments']}  experiments {len(rows) - 1}  best {min(vals):.5f}")


if __name__ == "__main__":
    main()
