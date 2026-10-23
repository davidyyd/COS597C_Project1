"""
paths.py -- where everything lives. Set by the repository's env.sh (AR_REPO, AR_DATA, AR_WORK); the defaults
below are the same as env.sh's, so a module imported without it still finds a fresh clone's layout.

  AR_DATA   the packed data and tokenizer (tokenizer/, train_rows.bin, val_rows.bin; see data/README)
  AR_WORK   everything a run writes:
              outputs/<date>/<tag>/        the agents' run folders
              traces/<date>/<tag>/         each run's record, outside the agent's reach (trace, requests, audit)
              state/enforced/              run settings, check and speed caches, model snapshots, agent configs
              raw_logs/enforced/           full training logs per experiment (never shown to agents)
              localgpu/                    the local GPU queue (common/localgpu.py)
"""

import os
import sys
from pathlib import Path

ARM = Path(__file__).resolve().parents[1]                     # .../enforced
REPO = Path(os.environ.get("AR_REPO") or ARM.parent).resolve()
DATA = Path(os.environ.get("AR_DATA") or REPO / "data" / "climbmix")
WORK = Path(os.environ.get("AR_WORK") or REPO / "work")
OUTPUTS, TRACES = WORK / "outputs", WORK / "traces"
STATE, RAW = WORK / "state" / "enforced", WORK / "raw_logs" / "enforced"
SNAPSHOTS = STATE / "smoke_models"                            # frozen copies of every checked model
ENV_SH = REPO / "env.sh"                                      # sourced by every GPU task

sys.path.insert(0, str(REPO / "common"))

TOKEN_FILE = Path.home() / ".ar_claude_token"                 # long-lived login from `claude setup-token`


def claude_auth():
    """environment variables that log a Claude Code process in, for processes with their own (empty) Claude
    config: the token in ~/.ar_claude_token (from `claude setup-token`; preferred), else ANTHROPIC_API_KEY.
    None if neither exists: then only an unjailed process with the user's own `claude` login can run."""
    if TOKEN_FILE.exists():
        return {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN_FILE.read_text().strip()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]}
    return None


def train_gpus():
    """GPUs for a training or speed benchmark: 8 (one node, as in our runs), or 4, 2 or 1 if there are fewer. The
    global batch is the same on any number (gradient accumulation), but the step time is not: on other
    hardware, measure the baseline and its caps again (tools/measure_baseline.py) before a run."""
    import localgpu
    n = localgpu.num_gpus()
    return next(g for g in (8, 4, 2, 1) if g <= max(n, 1))
