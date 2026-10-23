#!/usr/bin/env python3
"""
audit.py -- independent audit of a finished autoresearch run by a separate Claude session.

  python tools/audit.py DATE/TAG [--model claude-opus-5-5] [--effort high] [--prompt-only]

The auditor runs in the same kind of jail as the agents (tools/jail.py, role "audit"): it sees the run
folder, the run's trace store and the harness's raw logs of the run, all read-only, and writes only
$AR_WORK/traces/DATE/TAG/audit/ (audit.md, verdict.json and its helper files). Its own session
stream is recorded by this script outside that folder: $AR_WORK/traces/DATE/TAG/audit_trace.jsonl,
next to the exact prompt it got (audit_prompt.md). The checklist is template/audit.md. Without bubblewrap
(or with --no-jail) the auditor runs unjailed in that audit folder. Login as for the agents
(~/.ar_claude_token or ANTHROPIC_API_KEY; unjailed, also your own `claude` login).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jail  # noqa: E402

import paths  # noqa: E402

MASTER = jail.MASTER
TRACES, RAW = paths.TRACES, paths.RAW
SETTINGS = {"permissions": {"defaultMode": "dontAsk",
                            "allow": ["Bash", "Read", "Write", "Edit", "WebSearch", "WebFetch"]}}


def prompt_for(run):
    wt, tr, raw = jail.OUTPUTS / run, TRACES / run, RAW / run.replace("/", "__")
    rows = [r.split("\t") for r in (wt / "results.tsv").read_text().splitlines()[1:]]
    vals = [float(r[2]) for r in rows if len(r) > 2 and r[2] not in ("-", "")]
    state = json.loads((tr / "run.json").read_text())["state"]
    trace = tr / "trace.jsonl"
    with trace.open() as f:
        n_lines = sum(1 for _ in f)
    fill = dict(RUN=run, STRATEGY=state["strategy"], N_EXP=len(rows) - 1, BASELINE=f"{vals[0]:.5f}",
                BEST=f"{min(vals[1:] or vals):.5f}", RUN_DIR=wt,
                INSTRUCTIONS="AGENTS.md" if state.get("agent") == "codex" else "CLAUDE.md",
                TRACE_DIR=tr, TRACE_LINES=f"{n_lines:,}", TRACE_MB=f"{trace.stat().st_size / 1e6:.0f}",
                RAW_DIR=raw, CHECK_DIR=paths.STATE / "check_cache", OUT_DIR=tr / "audit")
    text = (MASTER / "template" / "audit.md").read_text()
    for k, v in fill.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def main():
    os.umask(0o077)                    # the audit's files: readable by the owner only
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--prompt-only", action="store_true", help="print the filled-in prompt and exit")
    ap.add_argument("--no-jail", action="store_true", help="run the auditor without the bubblewrap jail")
    a = ap.parse_args()
    tr = TRACES / a.run
    if not (tr / "trace.jsonl").exists():
        sys.exit(f"no trace for {a.run} in {tr}")
    prompt = prompt_for(a.run)
    if a.prompt_only:
        print(prompt)
        return
    use_jail = jail.available() and not a.no_jail
    auth = paths.claude_auth()
    if use_jail and not auth:
        sys.exit(f"no auditor login: run `claude setup-token` and save the token in {paths.TOKEN_FILE} (chmod 600), "
                 "or set ANTHROPIC_API_KEY")
    out = tr / "audit"
    out.mkdir(exist_ok=True)
    (tr / "audit_prompt.md").write_text(prompt)
    raw = RAW / a.run.replace("/", "__")
    checks = paths.STATE / "check_cache"
    if use_jail:
        env = jail.base_env(a.run, **auth)
        argv = jail.bwrap_argv(a.run, env, role="audit", cwd=out,
                               extra_ro=[tr] + [d for d in (raw, checks) if d.exists()],
                               extra_rw=[out]) + ["/ar/claude"]
        run_env = None
    else:
        print("WARNING: the auditor runs without the jail (it can read and write outside its audit folder)", flush=True)
        argv, run_env = [str(jail.claude_binary())], jail.unjailed_env(a.run, **(auth or {}))
    argv += ["-p", prompt, "--model", a.model, "--effort", a.effort, "--output-format", "stream-json",
             "--verbose", "--setting-sources", "project", "--settings", json.dumps(SETTINGS), "--strict-mcp-config"]
    print(f"[{datetime.now():%F %T}] auditing {a.run} ({a.model}, effort {a.effort}); stream: {tr / 'audit_trace.jsonl'}",
          flush=True)
    secrets = [v for k, v in (auth or {}).items() if k in jail.SECRET_VARS and v]
    with open(tr / "audit_trace.jsonl", "a") as f:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=run_env,
                           cwd=None if use_jail else out)
        for line in p.stdout:              # a login the auditor printed never reaches the trace
            line = line.decode(errors="replace")
            for v in secrets:
                line = line.replace(v, "<redacted>")
            f.write(line)
        p.wait()
    verdict = out / "verdict.json"
    print(f"[{datetime.now():%F %T}] auditor exited {p.returncode}; "
          + (f"verdict: {json.loads(verdict.read_text()).get('verdict')}; report: {out / 'audit.md'}"
             if verdict.exists() else "no verdict.json written"))


if __name__ == "__main__":
    main()
