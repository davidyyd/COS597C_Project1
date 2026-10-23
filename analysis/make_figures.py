#!/usr/bin/env python3
"""
make_figures.py -- the report's figure and table, recomputed from results/ (no GPUs needed).

  python analysis/make_figures.py            # writes analysis/climb_{all,zoom,legend}.pdf and analysis/summary.md

climb_*.pdf: best validation bits per byte so far against search round (100 rounds of 4 experiments) for the
            three setups; right panel zooms in on Enforced and Vanilla.
code_size.pdf / cost_score.pdf: lines changed vs the baseline by the best model so far, and each
            trained Enforced / Vanilla experiment's step time relative to the baseline against its score (Figure 3).
summary.md: per setup, the baseline, the best score and its experiment, how many experiments trained, and the
            post-run audit verdict (results/<setup>/audit/verdict.json).
"""

import csv
import difflib
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SETUPS = [("free", "#6b6b6b"), ("vanilla", "#c26a12"), ("enforced", "#2e6db4")]


def load(setup):
    rows = list(csv.DictReader(open(ROOT / "results" / setup / "results.tsv"), delimiter="\t"))
    key = "val_bpb" if "val_bpb" in rows[0] else "val_loss"      # the score is bits per byte in every setup
    base = float(rows[0][key])
    exps = []
    for r in rows[1:]:
        score = float(r[key]) if r["status"] == "trained" and r[key] not in ("", "-") else None
        exps.append(dict(name=r["name"], round=int(re.match(r"r(\d+)", r["name"]).group(1)), score=score))
    return base, exps


def climb(exps):
    """best score so far after each round; rounds are numbered by order of appearance"""
    order, best_in = [], {}
    for e in exps:
        if e["round"] not in best_in:
            order.append(e["round"])
            best_in[e["round"]] = float("inf")
        if e["score"] is not None:
            best_in[e["round"]] = min(best_in[e["round"]], e["score"])
    ys, best = [], float("inf")
    for k in order:
        best = min(best, best_in[k])
        ys.append(best)
    return ys


# Figure style of our ICLR 2027 paper (figures/plot_style.py there): one 2:1 panel per file, Arial at one size,
# thick curves, inward ticks, a full frame, no grid, and the group's palette (red, orange, blue).
COLOR = {"enforced": "#FF8988", "vanilla": "#FECC81", "free": "#6098FF"}
BASE_COLOR = "#9AA0A6"
FONTSIZE, LINEWIDTH, FIGSIZE = 32, 3.0, (12, 6)
AXES_RECT = (1.74 / 12, 0.185, 1 - 0.55 / 12 - 1.74 / 12, 1 - 1.0 / 6 - 0.185)


def apply_rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": FONTSIZE, "axes.labelsize": FONTSIZE, "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE, "legend.fontsize": FONTSIZE, "axes.titlesize": FONTSIZE,
        "axes.grid": False, "axes.linewidth": 1.5, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 8, "ytick.major.size": 8, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.major.pad": 9, "ytick.major.pad": 9, "axes.labelpad": 8, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 200})


def set_y(ax, ticks, fmt):
    """four evenly spaced ticks; the axis spans one step below the first to the last, and only the ticks at 50%
    and 100% of the height are drawn (as in the paper's panels)"""
    step = ticks[1] - ticks[0]
    ax.set_ylim(ticks[0] - step, ticks[-1])
    ax.set_yticks([ticks[1], ticks[3]], [fmt.format(ticks[1]), fmt.format(ticks[3])])


def panel(curves, base, setups, yticks, yfmt, title, path):
    apply_rc()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AXES_RECT)
    ax.axhline(base, color=BASE_COLOR, ls="--", lw=LINEWIDTH * 0.6, zorder=1)
    for setup in setups:
        ys = curves[setup]
        ax.step(range(len(ys)), ys, where="post", color=COLOR[setup], lw=LINEWIDTH, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    set_y(ax, yticks, yfmt)
    ax.set_xlabel("search round")
    ax.set_ylabel("best val bpb")
    ax.set_title(title, pad=20, fontsize=FONTSIZE * 1.1, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=100)     # for the README
    plt.close(fig)


def legend(path):
    apply_rc()
    fig = plt.figure(figsize=(FIGSIZE[0] * 2, 0.55))
    handles = [plt.Line2D([], [], color=COLOR[s], lw=LINEWIDTH, label=s.capitalize()) for s in ("free", "vanilla", "enforced")]
    handles.append(plt.Line2D([], [], color=BASE_COLOR, lw=LINEWIDTH * 0.6, ls="--", label="Baseline"))
    spread = 0.62
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False, handlelength=1.6, handletextpad=0.5,
               borderaxespad=0, mode="expand", bbox_to_anchor=((1 - spread) / 2, 0.0, spread, 1.0))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)


# Figure 3. Step time relative to the baseline: Enforced's benchmarked step time over the baseline's 65.6 ms;
# Vanilla's and Free's training time over the baseline's 249.2 s (every run trains the same 3,815 steps).
BASE_COST = {"enforced": 65.6, "vanilla": 249.2, "free": 249.2}
CODE = {"enforced": ("model.py", "baseline_model.py"), "vanilla": ("train.py", "baseline_train.py"),
        "free": ("train.py", "baseline_train.py")}


def _code(path):
    return [l.rstrip() for l in Path(path).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


def experiments(setup):
    """per experiment, in run order: round index (1..100), lines changed vs the baseline, step time / baseline, score"""
    res = ROOT / "results" / setup
    fn, base_fn = CODE[setup]
    base = _code(res / base_fn)
    rows = list(csv.DictReader(open(res / "results.tsv"), delimiter="\t"))[1:]
    key = "val_bpb" if "val_bpb" in rows[0] else "val_loss"
    rounds, out = {}, []
    for r in rows:
        k = int(re.match(r"r(\d+)", r["name"]).group(1))
        rounds.setdefault(k, len(rounds) + 1)
        code = res / "experiments" / r["name"] / fn
        if not code.exists():
            continue
        changed = sum(1 for l in difflib.unified_diff(base, _code(code), lineterm="", n=0)
                      if l[:1] in "+-" and not l.startswith(("+++", "---")))
        cost = score = None
        if r["status"] == "trained":
            score = float(r[key])
            if setup == "enforced":
                cost = json.loads((res / "experiments" / r["name"] / "result.json").read_text())["step_ms"]
            else:
                cost = float(r["training_seconds"])
            cost /= BASE_COST[setup]
        out.append(dict(round=rounds[k], changed=changed, cost=cost, score=score))
    return out


def code_panel(data, path):
    """lines changed vs the baseline by the best model found so far, after each round"""
    apply_rc()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AXES_RECT)
    for setup in ("free", "vanilla", "enforced"):
        best, xs, ys = None, [0], [0]
        for k in range(1, 101):
            for e in data[setup]:
                if e["round"] == k and e["score"] is not None and (best is None or e["score"] < best["score"]):
                    best = e
            xs.append(k)
            ys.append(best["changed"] if best else 0)
        ax.step(xs, ys, where="post", color=COLOR[setup], lw=LINEWIDTH)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    set_y(ax, [100, 200, 300, 400], "{:g}")
    ax.set_xlabel("search round")
    ax.set_ylabel("lines changed")
    ax.set_title("Code Size of the Best Model", pad=20, fontsize=FONTSIZE * 1.1, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)


def cost_panel(data, path):
    """Enforced and Vanilla only: step time relative to the baseline against score; the dashed line is the +5% limit
    both are given (Free trains 13x longer and is off this scale)"""
    apply_rc()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AXES_RECT)
    ax.axvline(1.05, color=BASE_COLOR, ls="--", lw=LINEWIDTH * 0.6, zorder=1)
    for setup in ("vanilla", "enforced"):
        pts = [(e["cost"], e["score"]) for e in data[setup] if e["score"] is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color=COLOR[setup], alpha=0.75,
                   linewidths=0, zorder=2)
    ax.set_xlim(0.8, 1.25)
    ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2], ["0.8", "0.9", "1.0", "1.1", "1.2"])
    set_y(ax, [0.929, 0.941, 0.953, 0.965], "{:.3f}")
    ax.set_xlabel("step time / baseline")
    ax.set_ylabel("val bpb")
    ax.set_title("Step Time vs. Score", pad=20, fontsize=FONTSIZE * 1.1, fontweight="bold")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)


def legend3(path):
    apply_rc()
    fig = plt.figure(figsize=(FIGSIZE[0] * 2, 0.55))
    handles = [plt.Line2D([], [], color=COLOR[s], lw=LINEWIDTH, label=s.capitalize()) for s in ("free", "vanilla", "enforced")]
    handles.append(plt.Line2D([], [], color=BASE_COLOR, lw=LINEWIDTH * 0.6, ls="--", label="+5% limit"))
    spread = 0.62
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False, handlelength=1.6, handletextpad=0.5,
               borderaxespad=0, mode="expand", bbox_to_anchor=((1 - spread) / 2, 0.0, spread, 1.0))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)


def main():
    out = ROOT / "analysis"
    curves, lines = {}, ["| Setup | Baseline | Best (bits/byte) | Change | Best experiment | Trained | Audit verdict |",
                         "|---|---|---|---|---|---|---|"]
    base = None
    for setup, _ in SETUPS:
        base, exps = load(setup)
        curves[setup] = [base] + climb(exps)
        trained = [e for e in exps if e["score"] is not None]
        best = min(trained, key=lambda e: e["score"])
        verdict = json.loads((ROOT / "results" / setup / "audit" / "verdict.json").read_text())["verdict"]
        lines.append(f"| {setup.capitalize()} | {base:.4f} | {best['score']:.4f} | {100 * (best['score'] / base - 1):+.1f}% "
                     f"| `{best['name']}` | {len(trained)}/{len(exps)} | {verdict} |")
    panel(curves, base, ("free", "vanilla", "enforced"), [0.80, 0.86, 0.92, 0.98], "{:.2f}", "All Setups",
          out / "climb_all.pdf")
    panel(curves, base, ("vanilla", "enforced"), [0.929, 0.941, 0.953, 0.965], "{:.3f}", "Enforced vs. Vanilla",
          out / "climb_zoom.pdf")
    legend(out / "climb_legend.pdf")
    data = {setup: experiments(setup) for setup, _ in SETUPS}
    code_panel(data, out / "code_size.pdf")
    cost_panel(data, out / "cost_score.pdf")
    legend3(out / "fig3_legend.pdf")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
