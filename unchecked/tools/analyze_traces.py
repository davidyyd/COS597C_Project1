#!/usr/bin/env python3
"""
analyze_traces.py -- trace analysis of autoresearch runs (in the spirit of Prime Intellect's
"Measuring autonomous research"). Read-only on the run folders.

usage:
  python tools/analyze_traces.py RUN_DIR [RUN_DIR ...] --out OUT_DIR
         [--raw-root $AR_WORK/raw_logs/<arm>] [--baseline VAL] [--idle-min 10]

(--baseline defaults to each run's own baseline score from its settings; the step-time cap is read
from harness/caps.json at the run's base commit, or 168 ms for runs before 2026-10-07.)

RUN_DIR is an agent run folder ($AR_WORK/outputs/<date>/<tag>/ with results.tsv, experiments/); its
trace is read from $AR_WORK/traces/<date>/<tag>/trace.jsonl. Raw experiment logs are read from
RAW_ROOT/<date>__<tag>/<name>/ (meta.json, and the task log of each training: run/task.log here, or
check.log and train/slurm.out in the enforced arm's layout); RAW_ROOT defaults to the run's arm's raw logs. Writes OUT_DIR/trace_metrics.json (all metrics, per run and pooled) and, if
matplotlib is importable, PNG figures in OUT_DIR/.

Needs python3 stdlib only; matplotlib is optional (figures are skipped without it).
Metric definitions are in the docstrings of the metric_* functions and in the report.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AR_WORK") or Path(os.environ.get("AR_REPO") or HERE.parents[1]) / "work")
GPUS_PER_TRAIN = 8

# --------------------------------------------------------------------------------------------
# idea families (metric C): ordered keyword table applied to the experiment's name stem.
# First matching row wins; names matching nothing fall back to "other:<first stem token>".
# --------------------------------------------------------------------------------------------
RERUN_RE = r"\b(rerun|replicate|retry|repeat|again|noise)\b"
FAMILIES = [
    ("ablation", r"^(ablate|ablation|remove|no|drop|without|undo|nothr)\b|\b(ablate|remove|without|undo)\b"),
    ("bilinear", r"bilinear|bil(\d|\b)"),
    ("successor/lookahead", r"succ|successor|lookahead"),
    ("induction/match", r"induction|match|idmask|pointer|recurrence"),
    ("copy/candidate", r"copy|candidate|cand\b|vote"),
    ("n-gram/bigram", r"bigram|trigram|ngram|skip-gram|transition"),
    ("surprise/pred-error", r"surprise|error|expectation|novel|unusual|predictab|confidence|belief|hypothesis"),
    ("readout/lens/head-rows", r"readout|lens|logit|readback|reread|head-?rows?|\brows?\b|lse|head-only|output-row"),
    ("memory/cache/gist", r"memory|cache|gist|dictionary|recall"),
    ("counts/occurrence", r"count|counter|occurrence|repetition|freq|first-mention"),
    ("lexical/identity-codes", r"lexical|\blex|identity|codes?\b|tokentopic|token-topic"),
    ("temperature/sharpness", r"temperature|temp\b|sharpness|concentration|salience|softcap"),
    ("document/position", r"doc|document|sentence|paragraph|line|position|phase|rope|rotary|clock|span|nesting|recency|distance|nope|radial"),
    ("stability/depth/carry", r"stability|depth|settled|settledness|curvature|carry|highway|crosslayer|cross-depth"),
    ("thresholds", r"thresh|\bthr\d*\b|threshold"),
    ("gating/gains", r"gate|gating|gains?\b|polarity"),
    ("tying", r"\btie\b|tied|tying"),
    ("mlp/neurons", r"mlp|neuron|habituation|adaptation|detector"),
    ("embeddings", r"embed|emb\b|table"),
    ("efficiency", r"fp8|fused|fast|lean|refactor|consolidated|speed|cheap"),
    ("attention-misc", r"quer|key|value|head|attn|attention|\bkv\b"),
]

# --------------------------------------------------------------------------------------------
# Bash command classes (metric E): first matching row wins.
# --------------------------------------------------------------------------------------------
BASH_CLASSES = [
    ("ar-exp (submit)", r"(^|[\s;&|(])ar-exp\b"),
    ("ar-check", r"(^|[\s;&|(])ar-check\b"),
    ("ar-gpu (own GPU job)", r"(^|[\s;&|(])ar-gpu\b"),
    ("wait/poll", r"\bsleep\b|\buntil\b|tasks/[^\s]*\.output|tail -f|\bsqueue\b|\bwait\b"),
    ("diff", r"(^|[;&|]\s*|\n)\s*(git )?(diff|cmp)\s+(-|<\(|[\w./$-]+\.(py|md|txt|tsv))"),
    ("python edit (patch script)", r"python[0-9.]*\b[^\n]*(apply_|edit_|mk\.py)|write_text|open\([^)]*['\"]w['\"]|\.replace\("),
    ("python analysis", r"\bpython[0-9.]*\b"),
    ("file ops (cp/mv/mkdir/sed -i/rm)", r"^\s*(cp|mv|mkdir|rm|rmdir|chmod|tee)\b|sed -i|cat >|\bcp\b .*model\.py"),
    ("read results/notes/logs", r"results\.tsv|result\.json|train\.log|notes\.md|check\.txt|idea\.md|error\.txt"),
    ("inspect code/files (cat/grep/ls)", r"^\s*(cat|grep|ls|head|tail|wc|awk|sed|find|for|echo|cd|pwd|paste)\b"),
]
POLL_TOOLS = {"Monitor", "TaskOutput", "ScheduleWakeup", "BashOutput"}


# ============================================================================================
# loading
# ============================================================================================
def parse_ts(s):
    """ISO time (with Z or offset) -> epoch seconds"""
    if s is None:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def code_file(d):
    """the experiment's code: model.py (enforced arm) or train.py (vanilla / free)"""
    return d / "model.py" if (d / "model.py").exists() else d / "train.py"


def baseline_file(run_dir):
    return run_dir / "baseline_model.py" if (run_dir / "baseline_model.py").exists() else run_dir / "baseline_train.py"


def sha(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def train_intervals(slurm_out):
    """[(start, end)] of the training attempts in a task log (start=/end= lines)"""
    try:
        txt = Path(slurm_out).read_text(errors="replace")
    except OSError:
        return []
    starts = [parse_ts(m) for m in re.findall(r"\bstart=(\S+)", txt)]
    ends = [parse_ts(m) for m in re.findall(r"^end=(\S+)", txt, re.M)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else mtime(slurm_out)   # killed job: last write of the log
        if s and e and e >= s:
            out.append((s, e))
    return out


def load_experiments(run_dir, raw_dir, baseline_val):
    """one dict per experiment folder (incl. ones killed before a result), plus results.tsv order"""
    rows = []
    with open(run_dir / "results.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            r = dict(zip(header, parts))
            if r.get("status") == "baseline":
                continue
            rows.append(r)
    order = {r["name"]: i for i, r in enumerate(rows)}
    exps = {}
    for d in sorted((run_dir / "experiments").iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        res = read_json(d / "result.json")
        raw = raw_dir / name
        meta = read_json(raw / "meta.json") or {}
        idea = (d / "idea.md").read_text(errors="replace") if (d / "idea.md").exists() else ""
        code = code_file(d)
        e = dict(name=name, idx=order.get(name), status=(res or {}).get("status", "no-result (killed)"),
                 val=(res or {}).get("val_loss", (res or {}).get("val_bpb")), step_ms=(res or {}).get("step_ms"),
                 reason=(res or {}).get("reason"), hash=sha(code), idea=idea,
                 title=idea.strip().splitlines()[0].lstrip("# ").strip() if idea.strip() else "",
                 submitted=meta.get("submitted") or mtime(code),
                 result_time=mtime(d / "result.json") if res else None)
        # check job (1 GPU): header time -> check.json write time
        cl = raw / "check.log"
        cs = None
        if cl.exists():
            m = re.search(r"at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) =====", cl.read_text(errors="replace")[:500])
            if m:
                cs = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()   # local time
        ce = mtime(raw / "check.json") or mtime(cl)
        e["check_iv"] = (cs, ce) if cs and ce and ce >= cs else None
        ivs = []
        for sub in ("train", "train_retry", "run", "run_retry"):
            for log in ("slurm.out", "task.log"):
                ivs += train_intervals(raw / sub / log)
        e["train_ivs"] = ivs
        e["train_s"] = sum(b - a for a, b in ivs)
        # end_time: when the agent could see the result (result.json written); compute_end: when the
        # last GPU job of the experiment ended (differs when ar-exp wrote results late, e.g. after an outage)
        ends = [b for _, b in ivs] + ([e["check_iv"][1]] if e["check_iv"] else [])
        e["compute_end"] = max(ends) if ends else e["result_time"]
        e["end_time"] = e["result_time"] if e["result_time"] else e["compute_end"]
        exps[name] = e
    return exps, rows


def trace_segments(run_dir):
    """(segment name, lines) of a run's trace: $AR_WORK/traces/<date>/<tag>/trace.jsonl, split at the
    launcher's {"type": "segment"} marker lines (one per session segment)"""
    date, tag = run_dir.parts[-2:]
    segs = []
    for line in open(WORK / "traces" / date / tag / "trace.jsonl"):
        if line.startswith('{"type": "segment"'):
            segs.append((f"segment_{json.loads(line)['n']:03d}.jsonl", []))
        elif segs:
            segs[-1][1].append(line)
    return segs


def load_trace(run_dir):
    """events of all segments: tool calls, timestamps, token accounting inputs"""
    segs = []
    for name, lines in trace_segments(run_dir):
        seg = dict(file=name, events=[], results=[], tools=[], first=None, last=None, tool_end={})
        last_ts = None
        pend_think = 0
        for line in lines:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            ts = parse_ts(ev.get("timestamp")) if ev.get("timestamp") else None
            if ts:
                last_ts = ts
                seg["first"] = seg["first"] or ts
                seg["last"] = ts
            if t == "system" and ev.get("subtype") == "thinking_tokens":
                pend_think += ev.get("estimated_tokens_delta", 0) or 0
            elif t == "assistant":
                chars = 0
                sub = bool(ev.get("parent_tool_use_id"))
                for b in ev["message"].get("content", []):
                    if b.get("type") == "text":
                        chars += len(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        chars += len(json.dumps(b.get("input", {})))
                        seg["tools"].append(dict(ts=ts or last_ts, name=b.get("name"), input=b.get("input", {}),
                                                 sub=sub, id=b.get("id"), seg=seg["file"]))
                seg["events"].append(dict(ts=ts or last_ts, chars=chars, think=pend_think))
                pend_think = 0
            elif t == "user" and ts:
                for b in (ev.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        seg["tool_end"][b.get("tool_use_id")] = ts
            elif t == "result":
                u = ev.get("usage") or {}
                seg["results"].append(dict(pos=len(seg["events"]), out=u.get("output_tokens", 0) or 0,
                                           think=(u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                                           is_error=ev.get("is_error"), text=str(ev.get("result", ""))[:200],
                                           cost=ev.get("total_cost_usd")))
        if pend_think and seg["events"]:
            seg["events"][-1]["think"] += pend_think
        segs.append(seg)
    return segs


# ============================================================================================
# A. output tokens over time
# ============================================================================================
def calibrate(seg_lists):
    """chars of visible output (text + tool inputs) per output token, from turns closed by a result
    event with nonzero usage: sum(chars) / sum(result output_tokens - thinking estimate)"""
    num = den = 0.0
    for segs in seg_lists:
        for s in segs:
            start = 0
            for r in s["results"]:
                evs = s["events"][start:r["pos"]]
                if r["out"] > 0 and evs:
                    th = sum(e["think"] for e in evs)
                    if r["out"] > th:
                        num += sum(e["chars"] for e in evs)
                        den += r["out"] - th
                    start = r["pos"]
    return num / den if den else 2.0


def token_timeline(segs, c):
    """Output tokens per assistant event. Turns closed by a `result` event with nonzero usage get
    that result's output_tokens, spread over the turn's assistant events in proportion to
    (thinking-token estimate + visible chars / c). Events not covered by any such result (a
    segment killed mid-turn, the 12 h segment limit) get the estimate itself, with c (chars per
    output token of visible text) calibrated on the covered turns."""
    pts = []          # (ts, tokens)
    per_seg = []
    for s in segs:
        start = 0
        covered = est = 0.0
        for r in s["results"]:
            evs = s["events"][start:r["pos"]]
            if r["out"] > 0 and evs:
                w = [e["think"] + e["chars"] / c for e in evs]
                tw = sum(w) or 1.0
                for e, wi in zip(evs, w):
                    pts.append((e["ts"], r["out"] * wi / tw))
                covered += r["out"]
                start = r["pos"]
        for e in s["events"][start:]:
            v = e["think"] + e["chars"] / c
            pts.append((e["ts"], v))
            est += v
        per_seg.append(dict(segment=s["file"], from_result_events=round(covered), estimated=round(est),
                            first=s["first"], last=s["last"]))
    pts = sorted((t, v) for t, v in pts if t)
    return pts, per_seg, c


# ============================================================================================
# helpers
# ============================================================================================
def best_so_far_events(exps):
    """experiments with a result, sorted by result time, with running best (trained only)"""
    done = sorted((e for e in exps.values() if e["result_time"]), key=lambda e: e["result_time"])
    best = None
    out = []
    for e in done:
        new = False
        prev = best[1] if best else None
        if e["status"] == "trained" and e["val"] is not None and (best is None or e["val"] < best[1]):
            best = (e["name"], e["val"])
            new = True
        out.append(dict(name=e["name"], t=e["result_time"], best=best[1] if best else None,
                        best_name=best[0] if best else None, new_best=new, prev_best=prev))
    return out


def stem(name):
    s = name.lower()
    s = re.sub(r"^(e\d+|r\d+[a-z]\d*)-", "", s)
    s = re.sub(r"^(r\d+[a-z]\d*|baseline|r\d+x)-", "", s)       # search: parent tag in the name
    return s


def family_of(name):
    s = stem(name)
    rerun = bool(re.search(RERUN_RE, s.replace("-", " ")))
    s2 = re.sub(r"-?(rerun|replicate|retry|repeat|again|noise|v\d+|fixed|fix|eps|fast|\d+)\b", "", s)
    words = s2.replace("-", " ").strip()
    for fam, rx in FAMILIES:
        if re.search(rx, words):
            return fam, rerun
    tok = (words.split() or [s or "?"])[0]
    return f"other:{tok}", rerun


def resolve_name(tok, names):
    if tok in names:
        return tok
    cands = [n for n in names if n.startswith(tok + "-")]
    return cands[0] if len(cands) == 1 else None


def parse_parent(e, names):
    m = re.search(r"^\**\s*parent\s*\**\s*:?\**\s*([A-Za-z0-9_+-]+)", e["idea"], re.I | re.M)
    if not m:
        return None, None, None
    tok = m.group(1).rstrip(".,;:")
    if tok.lower() == "baseline":
        par = "baseline"
    else:
        par = resolve_name(tok, names)
    after = e["idea"][m.start():m.start() + 400].lower()
    t = re.search(r"\b(exploit|explore)\b", after)
    return tok, par, (t.group(1) if t else None)


def bash_class(cmd):
    for cls, rx in BASH_CLASSES:
        if re.search(rx, cmd, re.M):
            return cls
    return "other"


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def fmt_t(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else None


# ============================================================================================
# analysis of one run
# ============================================================================================
def run_settings(run_dir):
    """the run's settings ($AR_WORK/state/<arm>/<date>__<tag>.json)"""
    for f in sorted((WORK / "state").glob(f"*/{run_dir.parent.name}__{run_dir.name}.json")):
        return json.loads(f.read_text())
    return {}


def step_cap(run_dir):
    """the run's step-time cap in ms: caps.json at its base commit (runs from 2026-10-07 on), else 168"""
    commit = run_settings(run_dir).get("base_commit")
    if commit:
        import subprocess
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "show",
                              f"{commit}:harness/caps.json"], capture_output=True, text=True).stdout
        if out:
            c = json.loads(out)
            return c.get("step_ms_cap") or c["step_time_ratio_cap"] * c["baseline_step_ms"]
    return 168.0


def analyze_run(run_dir, raw_root, baseline_val, idle_min, cpt=None, segs=None):
    run_dir = Path(run_dir).resolve()
    tag = f"{run_dir.parent.name}__{run_dir.name}"
    raw_root = raw_root or WORK / "raw_logs" / run_settings(run_dir).get("arm", "")
    exps, rows = load_experiments(run_dir, Path(raw_root) / tag, baseline_val)
    segs = segs if segs is not None else load_trace(run_dir)
    cpt = cpt or calibrate([segs])
    names = set(exps)
    tools = [t for s in segs for t in s["tools"]]
    trace_ts = sorted(t for s in segs for t in (s["first"], s["last"]) if t)
    ev_ts = sorted(e["ts"] for s in segs for e in s["events"] if e["ts"])
    t0 = trace_ts[0]
    # window: first to last agent event; GPU jobs still running or logging after the agent's last
    # event (orphaned ar-exp calls of killed sessions) are clipped to it
    t1 = trace_ts[-1]
    for e in exps.values():
        e["train_ivs"] = [(a_, min(b_, t1)) for a_, b_ in e["train_ivs"] if a_ < t1]
        e["train_s"] = sum(b_ - a_ for a_, b_ in e["train_ivs"])
        if e["check_iv"]:
            e["check_iv"] = (e["check_iv"][0], min(e["check_iv"][1], t1)) if e["check_iv"][0] < t1 else None
        if e["compute_end"]:
            e["compute_end"] = min(e["compute_end"], t1)
        if e["end_time"] and not e["result_time"]:
            e["end_time"] = min(e["end_time"], t1)
    R = dict(run=run_dir.name, path=str(run_dir), start=fmt_t(t0), end=fmt_t(t1),
             wall_h=round((t1 - t0) / 3600, 2))

    # ---------------- parents (search: idea.md; linear/batched: best model at submission) ----
    bsf = best_so_far_events(exps)
    is_search = any(re.search(r"^\**\s*parent", e["idea"], re.I | re.M) for e in exps.values())

    def best_at(t):
        cur = ("baseline", baseline_val)
        for b in bsf:
            if b["t"] <= t and b["best"] is not None and b["best"] < cur[1]:
                cur = (b["best_name"], b["best"])
        return cur

    for e in exps.values():
        if is_search:
            e["parent_tok"], e["parent"], e["kind"] = parse_parent(e, names)
        else:
            e["parent"] = best_at(e["submitted"])[0]
            e["kind"] = None
        pv = baseline_val if e["parent"] == "baseline" else (exps[e["parent"]]["val"] if e["parent"] in exps else None)
        e["parent_val"] = pv
        e["beats_parent"] = bool(e["status"] == "trained" and e["val"] is not None and pv is not None and e["val"] < pv)
    # batches = ar-exp calls (experiments submitted within 30 s of each other)
    subs = sorted((e for e in exps.values() if e["submitted"]), key=lambda e: e["submitted"])
    batches = []
    for e in subs:
        if batches and e["submitted"] - batches[-1][-1]["submitted"] < 30:
            batches[-1].append(e)
        else:
            batches.append([e])
    # keeps ("new bests"): per ar-exp call, the call's best trained experiment if it beats the best of
    # all earlier calls (= the keep rule of linear/batched; for search: a new best model of the run)
    keeps = []
    best_v = baseline_val
    for bt in batches:
        tr = [x for x in bt if x["status"] == "trained" and x["val"] is not None]
        if not tr:
            continue
        w = min(tr, key=lambda x: x["val"])
        if w["val"] < best_v:
            keeps.append(dict(name=w["name"], val=round(w["val"], 5), prev_best=round(best_v, 5),
                              gain=round(best_v - w["val"], 5),
                              hours=round(((w["result_time"] or w["end_time"]) - t0) / 3600, 2),
                              is_rerun_of_identical_model=False))
            best_v = w["val"]
    keep_names = {k["name"] for k in keeps}

    # ---------------- A. budget curves ----------------
    pts, per_seg, cpt = token_timeline(segs, cpt)
    cum_tok = []
    acc = 0.0
    for t, v in pts:
        acc += v
        cum_tok.append((t, acc))

    def tok_at(t):
        lo, hi = 0, len(cum_tok)
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_tok[mid][0] <= t:
                lo = mid + 1
            else:
                hi = mid
        return cum_tok[lo - 1][1] if lo else 0.0

    gpu_jobs = []   # (end, gpu_h)
    for e in exps.values():
        for a, b in e["train_ivs"]:
            gpu_jobs.append((b, GPUS_PER_TRAIN * (b - a) / 3600))
        if e["check_iv"]:
            gpu_jobs.append((e["check_iv"][1], (e["check_iv"][1] - e["check_iv"][0]) / 3600))
    gpu_jobs.sort()

    def gpu_at(t):
        return sum(g for b, g in gpu_jobs if b <= t)

    curve = []
    for i, b in enumerate(bsf):
        curve.append(dict(name=b["name"], n_exp=i + 1, hours=round((b["t"] - t0) / 3600, 3),
                          out_tokens=round(tok_at(b["t"])), gpu_h=round(gpu_at(b["t"]), 2),
                          best=min(b["best"], baseline_val) if b["best"] is not None else baseline_val,
                          new_best=b["new_best"]))
    train_gpu_h = sum(GPUS_PER_TRAIN * e["train_s"] / 3600 for e in exps.values())
    check_gpu_h = sum((e["check_iv"][1] - e["check_iv"][0]) / 3600 for e in exps.values() if e["check_iv"])
    final_best = min((e["val"] for e in exps.values() if e["status"] == "trained" and e["val"]), default=None)
    total_tok = acc
    tok_from_results = sum(s["from_result_events"] for s in per_seg)

    def best_at_budget(key, x):
        v = baseline_val
        for c in curve:
            if c[key] <= x and c["best"] is not None:
                v = min(v, c["best"])
        return round(v, 5)

    R["A_budget"] = dict(
        n_experiments=len(exps), n_with_result=len(bsf), final_best=final_best,
        total_output_tokens=round(total_tok), output_tokens_from_result_events=tok_from_results,
        output_tokens_estimated=round(total_tok - tok_from_results), chars_per_token_calibrated=round(cpt, 3),
        gpu_h_train=round(train_gpu_h, 1), gpu_h_check=round(check_gpu_h, 1),
        gpu_h_train_rejected_or_killed=round(sum(GPUS_PER_TRAIN * e["train_s"] / 3600 for e in exps.values()
                                                 if e["status"] != "trained"), 1),
        best_at={"hours": {str(h): best_at_budget("hours", h) for h in (6, 12, 24, 36, 48, 60)},
                 "experiments": {str(n): best_at_budget("n_exp", n) for n in (25, 50, 100, 186, 300, 388)},
                 "out_tokens_M": {str(m): best_at_budget("out_tokens", m * 1e6) for m in (0.5, 1, 1.5, 2, 3, 4)},
                 "gpu_h": {str(g): best_at_budget("gpu_h", g) for g in (50, 100, 200, 300, 400, 600)}},
        per_segment_tokens=[{**s, "first": fmt_t(s["first"]), "last": fmt_t(s["last"])} for s in per_seg],
        curve=curve)

    # ---------------- B. noise ----------------
    by_hash = defaultdict(list)
    for e in exps.values():
        if e["status"] == "trained" and e["val"] is not None and e["hash"]:
            by_hash[e["hash"]].append(e)
    bh = sha(baseline_file(run_dir))
    groups = []
    for h, es in by_hash.items():
        vals = [(x["name"], x["val"]) for x in sorted(es, key=lambda x: x["submitted"])]
        if h == bh:
            vals = [("baseline", baseline_val)] + vals
        if len(vals) >= 2:
            groups.append(vals)
    pairs = []
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pairs.append(dict(a=g[i][0], b=g[j][0], val_a=round(g[i][1], 5), val_b=round(g[j][1], 5),
                                  diff=round(abs(g[i][1] - g[j][1]), 5)))
    ss = dof = 0.0
    for g in groups:
        m = sum(v for _, v in g) / len(g)
        ss += sum((v - m) ** 2 for _, v in g)
        dof += len(g) - 1
    R["B_noise"] = dict(identical_groups=[[n for n, _ in g] for g in groups], pairs=pairs,
                        n_groups=len(groups), n_pairs=len(pairs), ss=ss, dof=dof,
                        sigma_run=round(math.sqrt(ss / dof), 5) if dof else None,
                        mean_abs_pair_diff=round(statistics.mean(p["diff"] for p in pairs), 5) if pairs else None,
                        max_abs_pair_diff=round(max(p["diff"] for p in pairs), 5) if pairs else None)
    earlier_hash = {}
    for e in sorted(exps.values(), key=lambda x: x["submitted"] or 0):
        if e["status"] == "trained":
            earlier_hash.setdefault(e["hash"], e["name"])
    for k in keeps:
        k["is_rerun_of_identical_model"] = earlier_hash.get(exps[k["name"]]["hash"]) != k["name"]
    R["B_keeps"] = keeps

    # ---------------- C. idea families / re-probing ----------------
    fam_att = defaultdict(list)
    first_by_hash = {}
    for e in sorted(exps.values(), key=lambda x: x["submitted"]):
        f, rr = family_of(e["name"])
        src = first_by_hash.get(e["hash"])
        if src is not None:                          # byte-identical rerun: family of the original
            f = src["family"]
            rr = True
        elif rr and e["parent"] in exps:             # rerun-named but changed: family of its parent
            f = exps[e["parent"]].get("family", f)
        e["family"], e["rerun_named"] = f, rr
        first_by_hash.setdefault(e["hash"], e)
        fam_att[f].append(e)
    reprobes, same_base_retries = [], 0
    for f, atts in fam_att.items():
        for i, a in enumerate(atts):
            prior = [p for p in atts[:i] if p["end_time"] and p["end_time"] <= a["submitted"]]
            if not prior:
                continue
            last = prior[-1]
            if last["beats_parent"]:
                continue
            if a["parent"] != last["parent"]:
                reprobes.append(dict(family=f, name=a["name"], failed_before=last["name"],
                                     old_parent=last["parent"], new_parent=a["parent"],
                                     all_prior_failed=not any(p["beats_parent"] for p in prior),
                                     beats_parent=a["beats_parent"], keep=a["name"] in keep_names))
            else:
                same_base_retries += 1
    fam_table = []
    for f, atts in sorted(fam_att.items(), key=lambda kv: -len(kv[1])):
        fam_table.append(dict(family=f, attempts=len(atts), trained=sum(a["status"] == "trained" for a in atts),
                              beats_parent=sum(a["beats_parent"] for a in atts),
                              keeps=sum(a["name"] in keep_names for a in atts),
                              best=min((a["val"] for a in atts if a["status"] == "trained" and a["val"]), default=None),
                              first=fmt_t(atts[0]["submitted"]), last=fmt_t(atts[-1]["submitted"])))
    single = [t for t in fam_table if t["attempts"] == 1]
    rp_dead = [r for r in reprobes if r["all_prior_failed"]]
    R["C_families"] = dict(
        n_families=len(fam_table), table=fam_table,
        reprobes=reprobes, n_reprobes=len(reprobes),
        n_reprobe_beats_parent=sum(r["beats_parent"] for r in reprobes),
        n_reprobe_keep=sum(r["keep"] for r in reprobes),
        n_reprobe_of_dead_family=len(rp_dead),
        n_reprobe_of_dead_family_beats_parent=sum(r["beats_parent"] for r in rp_dead),
        n_reprobe_of_dead_family_keep=sum(r["keep"] for r in rp_dead),
        same_base_retries_after_failure=same_base_retries,
        single_attempt_families=[t["family"] for t in single],
        n_single_attempt=len(single),
        n_single_attempt_failed=sum(1 for t in single if t["beats_parent"] == 0),
        keeps_by_family=Counter(e["family"] for e in exps.values() if e["name"] in keep_names).most_common())

    # ---------------- D. time allocation ----------------
    step = 60
    nb = int((t1 - t0) // step) + 1
    train_b = [False] * nb
    flight_b = [False] * nb
    active_b = [False] * nb

    def mark(arr, a, b):
        for k in range(max(0, int((a - t0) // step)), min(nb, int((b - t0) // step) + 1)):
            arr[k] = True

    for e in exps.values():
        for a, b in e["train_ivs"]:
            mark(train_b, a, b)
        if e["submitted"] and e["compute_end"]:
            mark(flight_b, e["submitted"], e["compute_end"])
    gap = idle_min * 60
    for a, b in zip(ev_ts, ev_ts[1:]):
        if b - a <= gap:
            mark(active_b, a, b)
    if ev_ts:
        mark(active_b, ev_ts[-1], ev_ts[-1])
    # a tool call keeps the agent busy until its result arrives; ar-check / ar-gpu calls are the
    # agent's own pre-flight checks and benchmarks on pool GPUs
    owncheck_b = [False] * nb
    tool_end = {k2: v for s in segs for k2, v in s["tool_end"].items()}
    for t in tools:
        te = tool_end.get(t["id"])
        if t["ts"] and te and te >= t["ts"]:
            mark(active_b, t["ts"], te)
            if t["name"] == "Bash" and re.search(r"\bar-(check|gpu)\b", t["input"].get("command", "")):
                mark(owncheck_b, t["ts"], te)
    # waiting tools (an `until ... sleep` or Monitor) keep the agent alive without timestamped events
    # while an experiment is in flight, so idle = nothing in flight and no agent event within idle_min
    cats = Counter()
    for k in range(nb):
        if train_b[k]:
            cats["training (>=1 8-GPU job)"] += 1
        elif flight_b[k]:
            cats["submitted, not training (check/queue)"] += 1
        elif owncheck_b[k]:
            cats["agent: own ar-check/ar-gpu job (no experiment in flight)"] += 1
        elif active_b[k]:
            cats["agent: thinking/editing/other tools (no experiment in flight)"] += 1
        else:
            cats["idle (no experiment, no agent event)"] += 1
    idle_gaps = []
    k = 0
    while k < nb:
        if not (train_b[k] or flight_b[k] or active_b[k]):
            j = k
            while j < nb and not (train_b[j] or flight_b[j] or active_b[j]):
                j += 1
            if (j - k) * step >= gap:
                idle_gaps.append(dict(start=fmt_t(t0 + k * step), end=fmt_t(t0 + j * step),
                                      hours=round((j - k) * step / 3600, 2)))
            k = j
        else:
            k += 1
    turn, overlap = [], 0
    for prev, nxt in zip(batches, batches[1:]):
        ends = [x["end_time"] for x in prev if x["end_time"]]
        if not ends:
            continue
        d = nxt[0]["submitted"] - max(ends)
        if d < 0:
            overlap += 1
        else:
            turn.append(d / 60)
    conc = []   # mean number of concurrently training experiments while >=1 trains
    cnt = [0] * nb
    for e in exps.values():
        for a, b in e["train_ivs"]:
            for kk in range(max(0, int((a - t0) // step)), min(nb, int((b - t0) // step) + 1)):
                cnt[kk] += 1
    busy = [c for c in cnt if c]
    R["D_time"] = dict(
        minutes={k2: v for k2, v in cats.items()}, fractions={k2: pct(v, nb) for k2, v in cats.items()},
        idle_gaps=idle_gaps, idle_gap_hours=round(sum(g["hours"] for g in idle_gaps), 2),
        n_batches=len(batches), median_turnaround_min=med(turn),
        p90_turnaround_min=round(sorted(turn)[int(0.9 * (len(turn) - 1))], 1) if turn else None,
        median_turnaround_min_excl_gaps=med([t for t in turn if t < 60]),
        n_turnaround_over_60min=sum(t >= 60 for t in turn), n_submitted_before_previous_finished=overlap,
        mean_concurrent_trainings_when_busy=round(sum(busy) / len(busy), 2) if busy else None,
        median_train_min=med([e["train_s"] / 60 for e in exps.values() if e["status"] == "trained" and e["train_s"]]),
        median_submit_to_result_min=med([(e["result_time"] - e["submitted"]) / 60 for e in exps.values()
                                         if e["result_time"] and e["status"] == "trained"]))

    # ---------------- E. agent behaviour ----------------
    tc = Counter((t["name"], "subagent" if t["sub"] else "main") for t in tools)
    bash = [t for t in tools if t["name"] == "Bash"]
    bcls = Counter()
    for t in bash:
        t["cls"] = bash_class(t["input"].get("command", ""))
        bcls[t["cls"]] += 1
    n_poll = bcls["wait/poll"] + sum(1 for t in tools if t["name"] in POLL_TOOLS)
    web = [dict(tool=t["name"], query=t["input"].get("query") or t["input"].get("url"),
                by="subagent" if t["sub"] else "main", time=fmt_t(t["ts"])) for t in tools
           if t["name"] in ("WebSearch", "WebFetch")]
    agents = [dict(time=fmt_t(t["ts"]), type=t["input"].get("subagent_type"), description=t["input"].get("description"),
                   prompt=(t["input"].get("prompt") or "")[:300]) for t in tools if t["name"] in ("Agent", "Task")]
    # helper scripts the agent wrote (not model variants): *.py/*.sh outside experiments/ and trace/
    helpers = []
    for p in sorted(run_dir.rglob("*")):
        rel = p.relative_to(run_dir)
        if p.suffix not in (".py", ".sh") or not p.is_file() or rel.parts[0] in ("experiments", "trace") \
                or "__pycache__" in rel.parts or p.name in ("model.py", "baseline_model.py", "train.py", "baseline_train.py"):
            continue
        txt = p.read_text(errors="replace")
        if re.search(r"^class GPT\b", txt, re.M):
            continue
        uses = [t for t in bash if p.name in t["input"].get("command", "")
                and re.search(r"\b(python[0-9.]*|bash|sh|ar-gpu)\b|\./" + re.escape(p.name), t["input"]["command"])]
        helpers.append(dict(file=str(rel), bytes=p.stat().st_size, created=fmt_t(mtime(p)), runs=len(uses)))
    bash_cmds_per_class_examples = {}
    for t in bash:
        bash_cmds_per_class_examples.setdefault(t["cls"], t["input"].get("command", "")[:160])
    R["E_behaviour"] = dict(
        tool_calls={f"{n} ({w})": c for (n, w), c in tc.most_common()},
        n_tool_calls=len(tools), n_bash=len(bash), bash_classes=dict(bcls.most_common()),
        bash_examples=bash_cmds_per_class_examples,
        n_bash_background=sum(1 for t in bash if t["input"].get("run_in_background")),
        poll_calls=n_poll, poll_calls_per_experiment=round(n_poll / max(1, len(exps)), 2),
        poll_calls_per_batch=round(n_poll / max(1, len(batches)), 2),
        tool_calls_per_experiment=round(len(tools) / max(1, len(exps)), 2),
        web=web, n_web=len(web), subagents=agents, n_subagents=len(agents),
        helper_scripts=helpers, n_helpers=len(helpers), n_helpers_run_0=sum(h["runs"] == 0 for h in helpers),
        n_helpers_run_1=sum(h["runs"] == 1 for h in helpers), n_helpers_run_2plus=sum(h["runs"] >= 2 for h in helpers),
        ar_gpu_calls_by_gpus=dict(Counter(int((re.search(r"--gpus[ =](\d+)", t["input"].get("command", "")) or [0, 1])[1])
                                          for t in bash if t["cls"] == "ar-gpu (own GPU job)")),
        ar_gpu_background_calls=sum(1 for t in bash if t["cls"] == "ar-gpu (own GPU job)" and t["input"].get("run_in_background")),
        ar_gpu_blocking_gpu_h=round(sum(
            int((re.search(r"--gpus[ =](\d+)", t["input"].get("command", "")) or [0, 1])[1])
            * (tool_end[t["id"]] - t["ts"]) / 3600 for t in bash
            if t["cls"] == "ar-gpu (own GPU job)" and not t["input"].get("run_in_background")
            and tool_end.get(t["id"]) and t["ts"]), 1),
        segments=len(segs), auth_failures=sum(1 for s in segs for r in s["results"] if "authenticate" in r["text"]))

    # ---------------- F. hygiene ----------------
    cap = step_cap(run_dir)
    rej = [e for e in exps.values() if e["status"] == "reject-step-time"]
    rej_ms = sorted(e["step_ms"] for e in rej if e["step_ms"])
    trained_ms = [e["step_ms"] for e in exps.values() if e["status"] == "trained" and e["step_ms"]]
    exp_names = {e["name"] for e in exps.values()}
    dup_hash = [g for g in R["B_noise"]["identical_groups"]]
    all_hash = defaultdict(list)
    for e in exps.values():
        if e["hash"]:
            all_hash[e["hash"]].append(e["name"])
    same_as_parent = [e["name"] for e in exps.values() if e["parent"] in exps and exps[e["parent"]]["hash"] == e["hash"]]
    outliers = [dict(name=e["name"], val=e["val"]) for e in exps.values()
                if e["status"] == "trained" and e["val"] and e["val"] > baseline_val + 0.1]
    # ar-check before ar-exp
    checks, subs_ = [], []
    for t in bash:
        cmd = t["input"].get("command", "")
        for m in re.finditer(r"\bar-check\s+([^\s;&|]+)", cmd):
            checks.append((t["ts"], m.group(1)))
        for m in re.finditer(r"\bar-exp\s+((?:[A-Za-z0-9_-]+=[^\s;&|]+\s*)+)", cmd):
            for a in m.group(1).split():
                n, pth = a.split("=", 1)
                subs_.append((t["ts"], n, pth))
    checked = 0
    last_sub = {}
    for s_ in subs_:                     # a name resubmitted after a failed ar-exp call: keep the last
        if s_[1] in exp_names:
            last_sub[s_[1]] = s_
    exp_subs = sorted(last_sub.values())
    for ts, n, pth in exp_subs:
        prev_same = max([t2 for t2, n2, p2 in exp_subs if p2 == pth and t2 < ts], default=t0 - 1)
        ok = any(prev_same < ct <= ts and (cp == pth or cp.rstrip("/") == pth or "$" in cp or "*" in cp
                                          or os.path.normpath(cp) == os.path.normpath(pth)) for ct, cp in checks)
        checked += ok
    hyg = dict(
        status_counts=dict(Counter(e["status"] for e in exps.values()).most_common()),
        step_time_rejects=len(rej), step_time_reject_rate=pct(len(rej), len(exps)),
        reject_step_ms=rej_ms, reject_within_2pct_of_cap=sum(1 for x in rej_ms if x <= cap * 1.02),
        reject_median_ms=med(rej_ms),
        trained_within_1pct_below_cap=sum(1 for x in trained_ms if x >= cap * 0.99),
        gpu_h_on_rejects=round(sum(GPUS_PER_TRAIN * e["train_s"] / 3600 for e in rej), 1),
        crashes=[e["name"] for e in exps.values() if e["status"].startswith("crash")],
        diverged=[e["name"] for e in exps.values() if e["status"] == "diverged"],
        trained_outliers=outliers,
        killed_no_result=[e["name"] for e in exps.values() if e["status"].startswith("no-result")],
        byte_identical_groups=[v for v in all_hash.values() if len(v) > 1],
        n_rerun_named=sum(e["rerun_named"] for e in exps.values()),
        identical_to_parent=same_as_parent,
        ar_check_calls=len(checks), ar_exp_submissions_in_trace=len(exp_subs),
        submissions_checked_before=checked, pct_checked_before=pct(checked, len(exp_subs)))
    if is_search:
        mism = dict(unresolved=[], parent_not_finished=[], name_tag_mismatch=[], closer_other_model=[])
        lines = {n: set(open(code_file(run_dir / "experiments" / n), errors="replace").read().splitlines())
                 for n in exps if code_file(run_dir / "experiments" / n).exists()}
        lines["baseline"] = set(baseline_file(run_dir).read_text().splitlines())

        def jac(a, b):
            return len(a & b) / max(1, len(a | b))

        for e in exps.values():
            if e["parent"] is None:
                mism["unresolved"].append(dict(name=e["name"], parent=e.get("parent_tok")))
                continue
            p = e["parent"]
            if p != "baseline" and (exps[p]["end_time"] or 1e18) > e["submitted"]:
                mism["parent_not_finished"].append(dict(name=e["name"], parent=p))
            m = re.match(r"^r\d+[a-z]\d*-(r\d+[a-z]\d*)-", e["name"])
            if m:
                tagp = m.group(1)
                if not (p == tagp or p.startswith(tagp + "-")):
                    mism["name_tag_mismatch"].append(dict(name=e["name"], name_tag=tagp, parent=p))
            if e["name"] in lines:
                me = lines[e["name"]]
                jp = jac(me, lines[p])
                prior = [o for o in exps.values() if o["name"] != e["name"] and o["end_time"]
                         and o["end_time"] <= e["submitted"] and o["name"] in lines]
                best_o = max(prior, key=lambda o: jac(me, lines[o["name"]]), default=None)
                if best_o is not None:
                    jo = jac(me, lines[best_o["name"]])
                    if jo - jp >= 0.10 and best_o["name"] != p:
                        mism["closer_other_model"].append(dict(name=e["name"], parent=p, jaccard_parent=round(jp, 3),
                                                               closest=best_o["name"], jaccard_closest=round(jo, 3)))
        hyg["search_parent_checks"] = {k2: v for k2, v in mism.items()}
        hyg["search_parent_check_counts"] = {k2: len(v) for k2, v in mism.items()}
    R["F_hygiene"] = hyg

    # ---------------- G. lineage ----------------
    def chain(n):
        out = []
        while n and n != "baseline" and n in exps and len(out) < 200:
            out.append(n)
            n = exps[n]["parent"]
        return out

    best_name = min((e for e in exps.values() if e["status"] == "trained" and e["val"]), key=lambda e: e["val"])["name"]
    ch = chain(best_name)
    lin = dict(best=best_name, winning_lineage=ch[::-1], depth=len(ch),
               distinct_parents=len({e["parent"] for e in exps.values() if e["parent"]}),
               top_parents=Counter(e["parent"] for e in exps.values() if e["parent"]).most_common(10))
    if is_search:
        kinds = defaultdict(list)
        for e in exps.values():
            k2 = e["kind"] or ("exploit" if e["parent"] not in (None, "baseline") else "explore")
            e["kind2"] = k2
            kinds[k2].append(e)
        lin["by_kind"] = {k2: dict(n=len(v), trained=sum(x["status"] == "trained" for x in v),
                                   beats_parent=sum(x["beats_parent"] for x in v),
                                   keeps=sum(x["name"] in keep_names for x in v),
                                   from_baseline=sum(x["parent"] == "baseline" for x in v),
                                   best=min((x["val"] for x in v if x["status"] == "trained" and x["val"]), default=None))
                          for k2, v in kinds.items()}
        lin["kind_unstated"] = sum(1 for e in exps.values() if not e["kind"])
        lin["winning_lineage_kinds"] = [exps[n]["kind2"] for n in ch[::-1]]
        roots = Counter()
        for e in exps.values():
            c2 = chain(e["name"])
            if c2:
                roots[c2[-1]] += 1
        lin["explore_roots_with_descendants"] = sum(1 for r2, c in roots.items() if c > 1 and exps[r2]["kind2"] == "explore")
        lin["winning_lineage_root"] = ch[-1] if ch else None
    R["G_lineage"] = lin

    # strip the big internals
    R["_experiments"] = [dict(name=e["name"], idx=e["idx"], status=e["status"], val=e["val"], step_ms=e["step_ms"],
                              submitted=fmt_t(e["submitted"]), result=fmt_t(e["result_time"]),
                              train_min=round(e["train_s"] / 60, 1), family=e["family"], parent=e["parent"],
                              kind=e.get("kind2"), beats_parent=e["beats_parent"], keep=e["name"] in keep_names)
                         for e in sorted(exps.values(), key=lambda e: e["submitted"] or 0)]
    R["_timeline"] = dict(t0=t0, step=step, train=[int(x) for x in cnt], flight=[int(x) for x in flight_b],
                          active=[int(x) for x in active_b], tokens=[(round((t - t0) / 3600, 4), round(v)) for t, v in cum_tok[::20]])
    return R


# ============================================================================================
# pooled noise + keep classification
# ============================================================================================
def pooled(runs):
    ss = sum(r["B_noise"]["ss"] for r in runs)
    dof = sum(r["B_noise"]["dof"] for r in runs)
    sigma = math.sqrt(ss / dof) if dof else None
    sd_diff = sigma * math.sqrt(2) if sigma else None
    for r in runs:
        cls = Counter()
        for k in r["B_keeps"]:
            x = k["gain"] / sd_diff if sd_diff else None
            k["gain_over_noise"] = round(x, 2) if x is not None else None
            k["class"] = "<1x" if x < 1 else ("1-2x" if x < 2 else ">2x")
            cls[k["class"]] += 1
        r["B_noise"]["keep_classes"] = {c: cls.get(c, 0) for c in ("<1x", "1-2x", ">2x")}
        r["B_noise"]["n_keeps"] = len(r["B_keeps"])
        late = [k for k in r["B_keeps"] if k["hours"] > 24]
        r["B_noise"]["keeps_after_24h"] = len(late)
        r["B_noise"]["keeps_after_24h_below_2x"] = sum(k["class"] != ">2x" for k in late)
        r["B_noise"]["gain_from_keeps_below_2x"] = round(sum(k["gain"] for k in r["B_keeps"] if k["class"] != ">2x"), 5)
    return dict(sigma_run=round(sigma, 5) if sigma else None, sd_of_difference=round(sd_diff, 5) if sd_diff else None,
                dof=dof, n_pairs=sum(r["B_noise"]["n_pairs"] for r in runs))


# ============================================================================================
# figures
# ============================================================================================
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def figures(runs, pool, out, baseline_val):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: figures skipped")
        return []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e6e6e6",
                         "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    files = []
    # A. budget curves
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    specs = [("hours", "wall-clock hours since start"), ("n_exp", "experiments with a result"),
             ("out_tokens", "cumulative agent output tokens (millions)"), ("gpu_h", "GPU-hours (8 × train time + checks)")]
    for ax, (key, lab) in zip(axs.flat, specs):
        for i, r in enumerate(runs):
            c = [p for p in r["A_budget"]["curve"] if p["best"] is not None]
            xs = [0] + [p[key] / (1e6 if key == "out_tokens" else 1) for p in c]
            ys = [baseline_val] + [p["best"] for p in c]
            short = r["run"].split("-")[0]
            ax.step(xs, ys, where="post", color=COLORS[i], lw=2, label=short)
            ax.annotate(f"{short} {ys[-1]:.4f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, -3 + 9 * (i - 1)),
                        fontsize=8, color="#52514e")
        ax.axhline(baseline_val, color="#999999", lw=1, ls="--")
        ax.set_xlabel(lab)
        ax.set_ylabel("best val_loss so far")
        ax.set_ylim(min(min(p["best"] for p in r["A_budget"]["curve"] if p["best"]) for r in runs) - 0.005, baseline_val + 0.004)
    axs[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Best-so-far val_loss under four budgets (lower is better; dashed = baseline)", fontsize=11)
    fig.tight_layout()
    p = out / "budget_curves.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(p.name)

    # B. keep gains vs noise
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sd = pool["sd_of_difference"]
    for i, r in enumerate(runs):
        ks = r["B_keeps"]
        ax.scatter([k["hours"] for k in ks], [k["gain"] / sd for k in ks], s=28, color=COLORS[i],
                   edgecolor="#fcfcfb", linewidth=1, label=r["run"].split("-")[0], zorder=3)
    ax.axhline(1, color="#999999", lw=1, ls="--")
    ax.axhline(2, color="#555555", lw=1, ls="--")
    ax.text(70.5, 1.0, "1× noise", fontsize=8, color="#52514e", va="center")
    ax.text(70.5, 2.0, "2× noise", fontsize=8, color="#52514e", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("hours since start")
    ax.set_ylabel(f"gain of new best / noise (SD of a difference = {sd:.5f})")
    ax.set_title("Size of each new best relative to measured run-to-run noise", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out / "keep_gains_vs_noise.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(p.name)

    # D. time allocation timeline
    fig, axs = plt.subplots(len(runs), 1, figsize=(11, 1.4 * len(runs) + 1), sharex=True, squeeze=False)
    for i, (ax, r) in enumerate(zip(axs[:, 0], runs)):
        tl = r["_timeline"]
        n = len(tl["train"])
        hrs = [k * tl["step"] / 3600 for k in range(n)]
        ax.fill_between(hrs, 0, tl["train"], step="post", color=COLORS[i], alpha=0.85, lw=0, label="trainings running")
        fl = [0.35 if (tl["flight"][k] and not tl["train"][k]) else 0 for k in range(n)]
        ax.fill_between(hrs, 0, fl, step="post", color="#bbbbbb", lw=0, label="submitted, not training")
        idle = [-0.35 if not (tl["train"][k] or tl["flight"][k] or tl["active"][k]) else 0 for k in range(n)]
        ax.fill_between(hrs, 0, idle, step="post", color="#e34948", lw=0, label="idle")
        ax.set_ylabel(r["run"].split("-")[0], rotation=0, ha="right", va="center")
        ax.set_yticks([0, 2, 4] if max(tl["train"]) > 1 else [0, 1])
        ax.set_ylim(-0.6, max(tl["train"]) + 0.5)
    axs[0, 0].legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    axs[-1, 0].set_xlabel("hours since start (bars: number of 8-GPU trainings running; red below 0: idle)")
    fig.suptitle("GPU occupancy over the run", fontsize=11)
    fig.tight_layout()
    p = out / "time_allocation.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(p.name)

    # E. bash command classes
    fig, ax = plt.subplots(figsize=(10, 4.2))
    classes = [c for c, _ in BASH_CLASSES] + ["other"]
    w = 0.27
    for i, r in enumerate(runs):
        vals = [r["E_behaviour"]["bash_classes"].get(c, 0) for c in classes]
        ax.barh([k + (i - 1) * w for k in range(len(classes))], vals, height=w - 0.03, color=COLORS[i],
                label=r["run"].split("-")[0])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Bash calls")
    ax.legend(frameon=False)
    ax.set_title("Bash commands by class (main agent + subagents)", fontsize=11)
    fig.tight_layout()
    p = out / "bash_classes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    files.append(p.name)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-root", help="raw experiment logs (default: $AR_WORK/raw_logs/<the run's arm>)")
    ap.add_argument("--baseline", type=float, help="baseline val loss (default: each run's own, from its settings)")
    ap.add_argument("--idle-min", type=float, default=10, help="agent silence (min) that counts as idle")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    traces = [load_trace(Path(r).resolve()) for r in a.runs]
    cpt = calibrate(traces)
    base = lambda r: a.baseline or run_settings(Path(r).resolve()).get("baseline_val") or 3.31628
    runs = [analyze_run(r, a.raw_root, base(r), a.idle_min, cpt, t) for r, t in zip(a.runs, traces)]
    a.baseline = a.baseline or base(a.runs[0])
    pool = pooled(runs)
    figs = figures(runs, pool, out, a.baseline)
    for r in runs:
        r["B_noise"].pop("ss", None)
        r["_timeline"].pop("t0", None)
    res = dict(generated=datetime.now().isoformat(timespec="seconds"), baseline=a.baseline, pooled_noise=pool,
               figures=figs, runs=runs)
    (out / "trace_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {out / 'trace_metrics.json'} and {len(figs)} figures")


if __name__ == "__main__":
    main()
