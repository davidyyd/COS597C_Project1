#!/usr/bin/env python3
"""
jail.py -- run an agent (or any command) in a bubblewrap jail that shows only its run folder.

  python tools/jail.py probe RUN                 # leak check: every forbidden access must fail
  python tools/jail.py shell RUN [-- CMD ...]    # a shell (or CMD) inside RUN's jail, for testing

RUN is DATE/TAG ($AR_WORK/outputs/DATE/TAG). Inside the jail (enforced by the kernel, for the agent
process, every tool it uses and every child process):
  * the filesystem is empty except: system dirs (read-only), the Python environment (read-only), the
    Claude Code binary (read-only), the run folder (read-write, but experiments/, results.tsv, .git,
    baseline_train.py, prepare.py and CLAUDE.md / AGENTS.md read-only), a home holding only this run's Claude
    config, and a /tmp kept with the run's agent state ($AR_WORK/state/agent/<run>/);
  * no GPU device files and no ssh / Slurm (stubs over their binaries, where they exist);
  * its own process namespace: other processes on the machine (other runs, GPU tasks) are invisible;
  * the network is the host's (the agent may use the internet);
  * ar-exp is a relay stub (tools/jail_bin/ar-relay): it writes a request into RUN/.ar/ and the launcher,
    outside the jail, runs the real command (see autoresearch.py Relay).
Memory, CPU and process limits come from a systemd --user scope around bwrap, where systemd --user exists.

The jail needs `bwrap` (bubblewrap) and unprivileged user namespaces. Without them (available() is False),
autoresearch.py runs the agent unjailed, with a warning; ar-exp still goes through the same relay.
"""

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

MASTER = Path(__file__).resolve().parents[1]                     # this arm's code (unchecked/)
REPO = Path(os.environ.get("AR_REPO") or MASTER.parent)
WORK = Path(os.environ.get("AR_WORK") or REPO / "work")
OUTPUTS = WORK / "outputs"
AGENT_STATE = WORK / "state" / "agent"              # per run: claude config (home/.claude) and tmp
PY_PREFIXES = sorted({sys.prefix, sys.base_prefix})  # the Python environment, mounted read-only
JAIL_BIN = MASTER / "tools" / "jail_bin"
HOME = os.path.expanduser("~")
TOKEN_FILE = Path(HOME) / ".ar_claude_token"         # long-lived agent login (claude setup-token), mode 600
LIMITS = ["MemoryMax=48G", "CPUQuota=1600%", "TasksMax=1024"]   # per agent
READ_ONLY_IN_RUN = ["experiments", "results.tsv", ".git", "baseline_train.py", "prepare.py", "CLAUDE.md", "AGENTS.md"]
STUBS_OVER = ["/usr/bin/ssh", "/usr/bin/scp", "/usr/bin/sbatch", "/usr/bin/srun", "/usr/bin/salloc",
              "/usr/bin/scancel", "/usr/bin/squeue", "/usr/bin/sacct", "/usr/bin/sinfo", "/usr/bin/scontrol"]


def claude_binary():
    p = shutil.which("claude") or f"{HOME}/.local/bin/claude"
    if not Path(p).exists():
        sys.exit("Claude Code not found: install it (https://claude.com/claude-code) so that `claude` is on PATH")
    return Path(p).resolve()


@functools.lru_cache(None)
def available():
    """bwrap is installed and can create a sandbox here"""
    if not shutil.which("bwrap"):
        return False
    return subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-pid", "true"],
                          capture_output=True).returncode == 0


@functools.lru_cache(None)
def systemd_scope():
    """systemd-run --user --scope works here (resource limits around the jail)"""
    if not shutil.which("systemd-run"):
        return False
    return subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "true"], capture_output=True).returncode == 0


SECRET_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")   # passed to the jailed command by file, not argv


def agent_dirs(run, role="agent"):
    d = AGENT_STATE / (run.replace("/", "__") + ("" if role == "agent" else f"__{role}"))
    return d / "claude", d / "tmp"


def agent_auth(cfg_dir):
    """the agent's login, as environment variables for its session: ~/.ar_claude_token (claude setup-token), else
    ANTHROPIC_API_KEY, else the operator's own Claude Code login (its credentials file, copied into the agent's
    separate config directory). Exits with instructions if there is none."""
    cfg_dir = Path(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        return dict(CLAUDE_CODE_OAUTH_TOKEN=TOKEN_FILE.read_text().strip())
    if os.environ.get("ANTHROPIC_API_KEY"):
        return dict(ANTHROPIC_API_KEY=os.environ["ANTHROPIC_API_KEY"])
    creds = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path(HOME) / ".claude") / ".credentials.json"
    if creds.exists():
        dst = cfg_dir / ".credentials.json"
        shutil.copy(creds, dst)
        dst.chmod(0o600)
        return {}
    sys.exit(f"no agent login: run `claude setup-token` and save the token in {TOKEN_FILE} (chmod 600), "
             "or set ANTHROPIC_API_KEY, or log in once with `claude`")


def bwrap_argv(run, env, limits=True, extra_ro=(), extra_rw=(), role="agent", cwd=None):
    """argv prefix that runs a command in RUN's jail with exactly the environment `env`. role "agent": the
    run's research agent (run folder writable except READ_ONLY_IN_RUN); any other role (e.g. "audit"):
    the run folder is entirely read-only, with its own Claude config and tmp"""
    wt = OUTPUTS / run
    cfg, tmp = agent_dirs(run, role)
    for d in (cfg, tmp) + ((wt / ".ar",) if role == "agent" else ()):
        d.mkdir(parents=True, exist_ok=True)
    a = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
         "--hostname", "agent", "--clearenv"]
    secrets = {k: str(v) for k, v in env.items() if k in SECRET_VARS}
    for k, v in env.items():
        if k not in secrets:
            a += ["--setenv", k, str(v)]
    for p in ("/usr", "/etc"):
        if os.path.exists(p):
            a += ["--ro-bind", p, p]
    for link, target in (("/bin", "usr/bin"), ("/sbin", "usr/sbin"), ("/lib", "usr/lib"), ("/lib64", "usr/lib64")):
        a += ["--symlink", target, link]
    a += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/run", "--dir", "/var/tmp"]
    a += ["--bind", str(tmp), "/tmp"]
    a += ["--tmpfs", HOME, "--bind", str(cfg), f"{HOME}/.claude"]
    a += ["--ro-bind", str(claude_binary()), "/ar/claude"]
    for p in PY_PREFIXES:
        if not p.startswith("/usr"):
            a += ["--ro-bind", p, p]
    a += ["--ro-bind", str(JAIL_BIN), "/ar/bin"]
    for p in STUBS_OVER:
        if os.path.exists(p):
            a += ["--ro-bind", str(JAIL_BIN / "blocked"), p]
    if role == "agent":
        a += ["--bind", str(wt), str(wt)]
        for name in READ_ONLY_IN_RUN:
            if (wt / name).exists():
                a += ["--ro-bind", str(wt / name), str(wt / name)]
    else:
        a += ["--ro-bind", str(wt), str(wt)]
    for p in extra_ro:
        a += ["--ro-bind", str(p), str(p)]
    for p in extra_rw:
        a += ["--bind", str(p), str(p)]
    if secrets:                      # never on a command line (ps / pgrep -a show it): owner-only files, read by
        sdir = cfg.parent / "secrets"  # jail_bin/ar-secrets into the environment of the command it starts
        sdir.mkdir(mode=0o700, exist_ok=True)
        for k, v in secrets.items():
            f = sdir / k
            fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.write(fd, v.encode())
            os.close(fd)
            a += ["--ro-bind", str(f), f"/run/ar_secrets/{k}"]
    a += ["--chdir", str(cwd or wt), "--"] + (["/ar/bin/ar-secrets"] if secrets else [])
    if limits and systemd_scope():
        a = ["systemd-run", "--user", "--scope", "--quiet", *[f"-p{x}" for x in LIMITS], *a]
    return a


def base_env(run, **extra):
    return dict(HOME=HOME, USER=os.environ.get("USER", ""), LANG="C.UTF-8", TERM="dumb", TMPDIR="/tmp",
                PATH=f"/ar/bin:{sys.prefix}/bin:/usr/local/bin:/usr/bin", AR_RUN=run, AR_RUN_DIR=str(OUTPUTS / run),
                PYTHONNOUSERSITE="1", DISABLE_AUTOUPDATER="1", **extra)


PROBE = r"""
fail=0
deny() { if eval "$2" >/dev/null 2>&1; then echo "LEAK   $1"; fail=1; else echo "ok     $1"; fi; }
allow() { if eval "$2" >/dev/null 2>&1; then echo "ok     $1"; else echo "BROKEN $1"; fail=1; fi; }
deny "read the infrastructure (harness, launcher)"     "ls $PROBE_MASTER/harness"
deny "read the repository's git history"               "git -C $PROBE_REPO log -1"
deny "read the trace store"                            "ls $PROBE_WORK/traces"
deny "read the run settings and raw logs"              "ls $PROBE_WORK/state || ls $PROBE_WORK/raw_logs"
deny "list other runs"                                 "test \$(ls $PROBE_WORK/outputs/* | grep -vc \"^$AR_RUN_TAG\$\\|^\$\\|:\$\") -gt 0"
deny "read ~/.ssh or other home files"                 "ls $HOME/.ssh || ls $HOME/.ar_claude_token"
deny "use a GPU (nvidia-smi)"                          "nvidia-smi -L"
deny "use a GPU (torch)"                               "python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'"
deny "ssh to another machine"                          "ssh -o BatchMode=yes localhost true"
deny "see other processes"                             "test \$(ls /proc | grep -c '^[0-9]') -gt 30"
deny "write results.tsv"                               "echo x >> results.tsv"
deny "write experiments/"                              "touch experiments/.probe"
deny "write git history"                               "touch .git/probe"
deny "write prepare.py"                                "echo >> prepare.py"
deny "install into the Python environment"             "touch $PROBE_PY/lib/probe"
deny "see the login on a command line (ps, pgrep -a)"  "cat /proc/[0-9]*/cmdline 2>/dev/null | tr '\0' '\n' | grep -q probe-x$((1+1))"
allow "login reaches the agent's environment"         "test \"\$CLAUDE_CODE_OAUTH_TOKEN\" = sk-ant-probe-x$((1+1))"
allow "write own files in the run folder"              "touch .probe_ok && rm .probe_ok"
allow "internet (huggingface.co)"                      "curl -sSf -o /dev/null --max-time 15 https://huggingface.co"
allow "python with torch (CPU)"                        "python -c 'import torch'"
allow "relay command on PATH"                          "command -v ar-exp"
deny "no ar-check in this run"                         "command -v ar-check"
exit $fail
"""


def probe(run, quiet=False):
    """run the leak check inside RUN's jail; True if every forbidden access failed and every needed one worked"""
    env = base_env(run, CLAUDE_CODE_OAUTH_TOKEN="sk-ant-probe-x2", AR_RUN_TAG=Path(run).name, PROBE_MASTER=MASTER, PROBE_REPO=REPO, PROBE_WORK=WORK,
                   PROBE_PY=sys.prefix)
    p = subprocess.run(bwrap_argv(run, env) + ["bash", "-c", PROBE], capture_output=True, text=True)
    if not quiet or p.returncode:
        print(p.stdout + p.stderr, end="")
    return p.returncode == 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("probe", "shell"):
        sys.exit(__doc__)
    if not available():
        sys.exit("bwrap is not available here (install bubblewrap, and allow unprivileged user namespaces)")
    run = sys.argv[2]
    if not (OUTPUTS / run).is_dir():
        sys.exit(f"no run folder {OUTPUTS / run}")
    if sys.argv[1] == "probe":
        sys.exit(0 if probe(run) else 1)
    cmd = sys.argv[4:] if sys.argv[3:4] == ["--"] else ["bash"]
    argv = bwrap_argv(run, base_env(run)) + cmd
    os.execvp(argv[0], argv)
