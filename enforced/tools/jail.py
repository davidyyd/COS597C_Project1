#!/usr/bin/env python3
"""
jail.py -- run an agent (or any command) in a bubblewrap jail that shows only its run folder.

  python tools/jail.py probe RUN                 # leak check: every forbidden access must fail
  python tools/jail.py shell RUN [-- CMD ...]    # a shell (or CMD) inside RUN's jail, for testing

RUN is DATE/TAG ($AR_WORK/outputs/DATE/TAG). Inside the jail (enforced by the kernel, for the agent
process, every tool it uses and every child process):
  * the filesystem is empty except: system dirs (read-only), the Python environment (read-only), the
    Claude Code binary (read-only), the run folder (read-write, but experiments/, results.tsv, .git,
    baseline_model.py and CLAUDE.md read-only), a home holding only this run's Claude config, and a
    /tmp kept with the run's agent state ($AR_WORK/state/enforced/agent/<run>/);
  * no GPU device files, no ssh or Slurm (stubs over their binaries);
  * its own process namespace: other processes on the machine (other runs, GPU tasks) are invisible;
  * the network is the host's (the agent may use the internet);
  * ar-check and ar-exp are relay stubs (tools/jail_bin/ar-relay): they write a request into
    RUN/.ar/ and the launcher, outside the jail, runs the real command (see autoresearch.py relay).
Memory, CPU and process limits come from a systemd --user scope around bwrap, where one is available.

The jail needs bubblewrap (`bwrap`). Without it (or with `autoresearch.py start --no-jail`) the agent runs
unjailed in its run folder: the relay stubs are still first on its PATH and ar-check / ar-exp still go
through the launcher, but nothing stops the agent from reading the harness or other files. Use that only
for trying things out.
"""

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

MASTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MASTER / "harness"))
import paths  # noqa: E402

OUTPUTS = paths.OUTPUTS
AGENT_STATE = paths.STATE / "agent"                # per run: claude config (home/.claude) and tmp
PYBIN = Path(sys.executable).parent                # the Python environment's bin/ (python, torchrun)
PY_MOUNTS = sorted({p for p in (Path(sys.prefix), Path(sys.base_prefix)) if not str(p).startswith("/usr")})
JAIL_BIN = MASTER / "tools" / "jail_bin"
AGENT_BIN = MASTER / "tools" / "agent_bin"
HOME = os.path.expanduser("~")
LIMITS = ["MemoryMax=48G", "CPUQuota=1600%", "TasksMax=1024"]   # per agent
READ_ONLY_IN_RUN = ["experiments", "results.tsv", ".git", "baseline_model.py", "CLAUDE.md"]
STUBS_OVER = ["/usr/bin/ssh", "/usr/bin/scp", "/usr/bin/sbatch", "/usr/bin/srun", "/usr/bin/salloc",
              "/usr/bin/scancel", "/usr/bin/squeue", "/usr/bin/sacct", "/usr/bin/sinfo", "/usr/bin/scontrol"]


def available():
    """bubblewrap is installed"""
    return shutil.which("bwrap") is not None


@lru_cache(maxsize=None)
def systemd_scope():
    """a systemd --user scope can be started here (for the resource limits)"""
    if not shutil.which("systemd-run"):
        return False
    return subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "true"],
                          capture_output=True).returncode == 0


def claude_binary():
    return Path(shutil.which("claude") or f"{HOME}/.local/bin/claude").resolve()


SECRET_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")   # passed to the jailed command by file, not argv


def agent_dirs(run, role="agent"):
    d = AGENT_STATE / (run.replace("/", "__") + ("" if role == "agent" else f"__{role}"))
    return d / "claude", d / "tmp"


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
    for p in PY_MOUNTS:
        a += ["--ro-bind", str(p), str(p)]
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
    """the whole environment of a jailed process"""
    return dict(HOME=HOME, USER=os.environ.get("USER", ""), LANG="C.UTF-8", TERM="dumb", TMPDIR="/tmp",
                PATH=f"/ar/bin:{PYBIN}:/usr/local/bin:/usr/bin", AR_RUN=run, AR_RUN_DIR=str(OUTPUTS / run),
                PYTHONNOUSERSITE="1", DISABLE_AUTOUPDATER="1", **extra)


def unjailed_env(run, **extra):
    """the environment of an agent that runs without the jail: the host's, with the relay stubs first on
    PATH (so ar-check / ar-exp still go through the launcher) and the GPUs hidden"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
    env.update(PATH=f"{JAIL_BIN}:{AGENT_BIN}:{PYBIN}:{os.environ['PATH']}", AR_RUN=run,
               AR_RUN_DIR=str(OUTPUTS / run), CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
               DISABLE_AUTOUPDATER="1", **extra)
    return env


PROBE = r"""
fail=0
deny() { if eval "$2" >/dev/null 2>&1; then echo "LEAK   $1"; fail=1; else echo "ok     $1"; fi; }
allow() { if eval "$2" >/dev/null 2>&1; then echo "ok     $1"; else echo "BROKEN $1"; fail=1; fi; }
deny "read the harness"                                "ls $AR_REPO/enforced/harness"
deny "read the repository's git history"               "git -C $AR_REPO log -1"
deny "read the trace store"                            "ls $AR_WORK/traces"
deny "read the check cache and run settings"           "ls $AR_WORK/state/enforced"
deny "list other runs"                                 "test \$(ls $AR_WORK/outputs/* | grep -vc \"^$AR_RUN_TAG\$\\|^\$\\|:\$\") -gt 0"
deny "read ~/.ssh or other home files"                 "ls $HOME/.ssh || ls $HOME/.ar_claude_token"
deny "use a GPU (nvidia-smi)"                          "nvidia-smi -L"
deny "use a GPU (torch)"                               "python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'"
deny "Slurm: sbatch"                                   "sbatch --wrap=true"
deny "ssh"                                             "ssh -o BatchMode=yes localhost true"
deny "see other processes"                             "test \$(ls /proc | grep -c '^[0-9]') -gt 30"
deny "write results.tsv"                               "echo x >> results.tsv"
deny "write experiments/"                              "touch experiments/.probe"
deny "write git history"                               "touch .git/probe"
deny "write baseline_model.py"                         "echo >> baseline_model.py"
deny "install into the Python environment"             "touch $PYENV/probe"
deny "see the login on a command line (ps, pgrep -a)"  "cat /proc/[0-9]*/cmdline 2>/dev/null | tr '\0' '\n' | grep -q probe-x$((1+1))"
allow "login reaches the agent's environment"         "test \"\$CLAUDE_CODE_OAUTH_TOKEN\" = sk-ant-probe-x$((1+1))"
allow "write own files in the run folder"              "touch .probe_ok && rm .probe_ok"
allow "internet (huggingface.co)"                      "curl -sSf -o /dev/null --max-time 15 https://huggingface.co"
allow "python with torch (CPU)"                        "python -c 'import torch'"
allow "relay commands on PATH"                         "command -v ar-exp && command -v ar-check"
exit $fail
"""


def probe(run, quiet=False):
    """run the leak check inside RUN's jail; True if every forbidden access failed and every needed one worked"""
    env = base_env(run, CLAUDE_CODE_OAUTH_TOKEN="sk-ant-probe-x2", AR_RUN_TAG=Path(run).name, AR_REPO=paths.REPO, AR_WORK=paths.WORK,
                   PYENV=Path(sys.prefix) / "bin")
    p = subprocess.run(bwrap_argv(run, env) + ["bash", "-c", PROBE], capture_output=True, text=True)
    if not quiet or p.returncode:
        print(p.stdout + p.stderr, end="")
    return p.returncode == 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("probe", "shell"):
        sys.exit(__doc__)
    if not available():
        sys.exit("bubblewrap (bwrap) is not installed: the jail is unavailable")
    run = sys.argv[2]
    if not (OUTPUTS / run).is_dir():
        sys.exit(f"no run folder {OUTPUTS / run}")
    if sys.argv[1] == "probe":
        sys.exit(0 if probe(run) else 1)
    cmd = sys.argv[4:] if sys.argv[3:4] == ["--"] else ["bash"]
    argv = bwrap_argv(run, base_env(run)) + cmd
    os.execvp(argv[0], argv)
