"""
review.py -- FROZEN. The complexity review of ar-check, for runs started with `--review` (the run's
state file has "review": true; other runs never call it).

A separate, stateless Claude call (no tools, a fixed prompt: template/review.md, a JSON schema) judges
the complete change of a candidate relative to harness/baseline_model.py -- always the baseline, never
the experiment it builds on -- and scores its complexity from 1 to 10. ar-check passes a model only
with complexity <= MAX_COMPLEXITY. The reviewer sees the code without comments or docstrings (exactly
what the check's model hash covers), the diff, a few mechanical statistics, and the agent's idea.md
(marked as the author's claim); it never sees results. It runs on the host, outside the agent's jail,
with its own login (~/.ar_claude_token or ANTHROPIC_API_KEY; see harness/paths.py) and an empty config
directory. Off unless a run is started with --review (our enforced runs did not use it).
"""

import ast
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
MASTER = HARNESS.parent
PROMPT = MASTER / "template" / "review.md"
MAX_COMPLEXITY = 4
MODEL, EFFORT = "claude-opus-5-5", "medium"
VERSION = "review-v1"        # part of the check's rules: change it when the prompt, model or cutoff changes
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["mechanisms", "n_mechanisms", "red_flags", "method_summary", "complexity", "reasons"],
          "properties": {
              "mechanisms": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                             "required": ["name", "formula", "purpose", "where"],
                             "properties": {k: {"type": "string"} for k in ["name", "formula", "purpose", "where"]}}},
              "n_mechanisms": {"type": "integer", "minimum": 0},
              "red_flags": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                            "required": ["flag", "where"],
                            "properties": {k: {"type": "string"} for k in ["flag", "where"]}}},
              "method_summary": {"type": "string"},
              "complexity": {"type": "integer", "minimum": 1, "maximum": 10},
              "reasons": {"type": "string"}}}


def enabled():
    """whether the run calling ar-check / ar-exp (AR_RUN, set by the relay) was started with --review"""
    run = os.environ.get("AR_RUN")
    sys.path.insert(0, str(HARNESS))
    import paths
    f = paths.STATE / f"{run.replace('/', '__')}.json" if run else None
    return bool(f and f.exists() and json.loads(f.read_text()).get("review"))


def strip(source):
    """source without comments and docstrings"""
    tree = ast.parse(source)
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def stats(base, cand, diff):
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    count = lambda pat, src: len(re.findall(pat, src))
    lits = lambda src: set(re.findall(r"(?<![\w.])\d+\.\d+(?:e-?\d+)?|(?<![\w.])\d{2,}(?![\w.])", src))
    kinds = [r"nn\.Parameter", r"nn\.Linear", r"nn\.Embedding"]
    nb = "/".join(str(count(k, base)) for k in kinds)
    nc = "/".join(str(count(k, cand)) for k in kinds)
    layer_cond = re.compile(r"\b(i|idx|layer|layer_idx)\s*(<|>|<=|>=|==|in)\b")
    return "\n".join([
        f"- lines added / removed: {len(added)} / {len(removed)} (of {len(base.splitlines())} baseline lines)",
        f"- nn.Parameter / nn.Linear / nn.Embedding: {nb} in the baseline, {nc} in the candidate",
        f"- numeric literals new in the candidate: {sorted(lits(cand) - lits(base))[:30]}",
        f"- added lines with a layer-index condition: {sum(1 for l in added if layer_cond.search(l))}",
    ])


def review(model_file, idea_file=None, timeout=900):
    """the reviewer's verdict (dict, with "passed") for a model file; raises RuntimeError if the reviewer
    could not be reached (the caller must not store that as the model's outcome)"""
    base = strip((HARNESS / "baseline_model.py").read_text())
    cand = strip(Path(model_file).read_text())
    diff = list(difflib.unified_diff(base.splitlines(), cand.splitlines(), "baseline", "candidate", lineterm="", n=3))
    if not diff:
        return dict(mechanisms=[], n_mechanisms=0, red_flags=[], method_summary="identical to the baseline",
                    complexity=1, reasons="no change", passed=True, max_complexity=MAX_COMPLEXITY, version=VERSION)
    idea = Path(idea_file).read_text() if idea_file and Path(idea_file).exists() else "(no idea.md next to the model file)"
    prompt = (PROMPT.read_text().replace("{BASELINE_CODE}", base).replace("{CANDIDATE_CODE}", cand)
              .replace("{DIFF}", "\n".join(diff)).replace("{STATS}", stats(base, cand, diff)).replace("{IDEA}", idea))
    sys.path.insert(0, str(MASTER / "tools"))
    from jail import claude_binary
    err = None
    for attempt in (1, 2):
        with tempfile.TemporaryDirectory() as cfg, tempfile.TemporaryDirectory() as cwd:
            sys.path.insert(0, str(HARNESS))
            import paths
            auth = paths.claude_auth()
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
            if auth:                 # its own empty config; without a token or key, the user's `claude` login
                env.update(CLAUDE_CONFIG_DIR=cfg, **auth)
            try:
                p = subprocess.run([str(claude_binary()), "-p", "--tools", "", "--json-schema", json.dumps(SCHEMA),
                                    "--model", MODEL, "--effort", EFFORT, "--output-format", "json",
                                    "--setting-sources", "project"],
                                   input=prompt, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)
                res = json.loads(p.stdout)
                v = res.get("structured_output") or json.loads(res["result"])
                assert isinstance(v.get("complexity"), int)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
                continue
        v.update(passed=v["complexity"] <= MAX_COMPLEXITY, max_complexity=MAX_COMPLEXITY, version=VERSION,
                 cost_usd=res.get("total_cost_usd"))
        return v
    raise RuntimeError(f"the complexity reviewer could not be reached ({err})")


def report(v):
    """the review as the agent sees it (ar-check output and check.txt)"""
    lines = [f"complexity review (whole change vs baseline_model.py): {v['complexity']}/10 "
             f"(pass <= {v['max_complexity']}): {'PASS' if v['passed'] else 'FAIL'}",
             f"  method: {v['method_summary']}"]
    for m in v.get("mechanisms", []):
        lines.append(f"  mechanism: {m['name']}: {m['formula']}")
    for f in v.get("red_flags", []):
        lines.append(f"  red flag: {f['flag']} ({f['where']})")
    lines.append(f"  reviewer: {v['reasons']}")
    return "\n".join(lines) + "\n"
