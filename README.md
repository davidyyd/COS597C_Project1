# Can we trust an AI agent that does research by itself?

COS 597C, Project 1 · Yida Yin · October 2026 · report: [`report/report.pdf`](report/report.pdf)

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) lets a coding agent improve a small
language-model training script on its own: edit, train for ~4 minutes, keep the change if validation bits per
byte (bpb) went down, repeat. We ran it with Claude (Opus 5.5) for 400 experiments in three setups that differ
only in how the rules are enforced, and audited every run with a second, independent Claude session.

| Setup | Rules told | Rules checked | Best bpb (baseline 0.9589) | Audit verdict |
|---|---|---|---|---|
| **Free** | none | none | 0.7598 (−20.8%) | clean (no rule to break; won by a ~1.9B-parameter model) |
| **Vanilla** | change only the model; +5% time / FLOPs / size; fixed precision | none | 0.9213 (−3.9%) | **serious**: switched to bf16 math against the rules |
| **Enforced** (ours) | same as Vanilla | before every training (`ar-check`) | 0.9287 (−3.2%) | minor: no rule broken |

<img src="analysis/climb_all.png" width="49%"> <img src="analysis/climb_zoom.png" width="49%">
<img src="analysis/climb_legend.png" width="98%">

## Repository layout

| path | what |
|---|---|
| `enforced/` | the Enforced setup: the agent edits only a model file; `ar-check` (code scan, precision trace, causality test, +5% caps on parameters / FLOPs / step time) gates every training. See [`enforced/README.md`](enforced/README.md). |
| `unchecked/` | the Vanilla and Free setups (shared code; `--arm vanilla\|free`): the agent edits the whole `train.py`, nothing is checked. See [`unchecked/README.md`](unchecked/README.md). |
| `common/localgpu.py` | a small local GPU queue (stands in for the cluster scheduler we used) |
| `data/` | download and preparation of the data (Karpathy's ClimbMix shards, tokenizer, packed rows), with checksums |
| `results/<setup>/` | our three runs: `results.tsv`, every experiment's code (`model.py` or `train.py`), `idea.md` and `result.json`, the agent's `notes.md` and `CLAUDE.md`, and the post-run audit (`audit/audit.md`, `verdict.json`) |
| `analysis/` | `make_figures.py`: the report's Figure 2 (`climb_*.pdf`, PNG copies for this README) and a summary table, from `results/` |
| `report/` | the report in the ICLR 2027 template (`report.tex`, `references.bib`, `report.pdf`; build: `cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report`) |
| `env.sh` | sourced by every command: your Python environment and where data and outputs go |

The raw session traces are not included (about 10 MB per run): they contain machine details and, in one case, a
login token that leaked into the agent's view (see the report, "What did not work"). The audits that cite them are
included.

## 1. Look at the results (no GPU needed)

```bash
git clone https://github.com/davidyyd/COS597C_Project1.git && cd COS597C_Project1
python3 -m pip install matplotlib
python3 analysis/make_figures.py                  # -> analysis/climb_{all,zoom,legend}.{pdf,png}, analysis/summary.md
less results/vanilla/audit/audit.md               # the audit that found the bf16 switch
column -t -s $'\t' results/enforced/results.tsv | sort -k3 -n | head    # the 10 best Enforced models
```

## 2. Setup (for running anything on GPUs)

Hardware: our runs used one node with 8 H100 (80 GB) GPUs per training (about 4 minutes each). Fewer or smaller
GPUs work (the global batch is kept with gradient accumulation), but each training takes longer, and the Enforced
step-time cap must then be re-measured on your hardware (`enforced/README.md`, "Hardware notes").

```bash
python3.11 -m venv ~/venvs/autoresearch && source ~/venvs/autoresearch/bin/activate
pip install -r requirements.txt
export AR_ACTIVATE="source ~/venvs/autoresearch/bin/activate"   # env.sh runs this in every GPU task
source env.sh                                                   # sets AR_REPO, AR_DATA, AR_WORK
```

Data (about 4.2 GB of downloads, about 4 GB prepared; preparation takes roughly 30-60 minutes on 32 CPU cores):

```bash
bash data/download.sh                    # 49 ClimbMix shards -> $AR_DATA/parquet/
python data/prepare_climbmix.py --workers 32   # tokenizer + packed train/val rows -> $AR_DATA/
(cd $AR_DATA && sha256sum -c $AR_REPO/data/SHA256SUMS.txt)   # should match ours exactly
```

## 3. Check and train the baseline

```bash
source env.sh
# the Enforced gate on the baseline model: model check on 1 GPU, then the speed benchmark on 8 GPUs
enforced/tools/bin/ar-check enforced/harness/baseline_model.py

# one full training of the baseline (2.0B tokens, about 4 minutes on 8 H100s); expect val bpb ~0.959
export GQ_TASK_ID=manual GQ_ATTEMPT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash enforced/harness/run_train.sh $AR_REPO/enforced/harness/baseline_model.py 1e-2 $AR_WORK/baseline_enforced
cat $AR_WORK/baseline_enforced/result.json

# the same baseline as a Vanilla/Free train.py
bash unchecked/harness/run_vanilla.sh $AR_REPO/unchecked/harness/vanilla/train.py $AR_WORK/baseline_vanilla
cat $AR_WORK/baseline_vanilla/result.json
unset GQ_TASK_ID GQ_ATTEMPT CUDA_VISIBLE_DEVICES
```

## 4. Run the agent

Needs [Claude Code](https://docs.claude.com/en/docs/claude-code) (the native installer, so that the jail can mount
its single binary) and a login: `claude setup-token`, then save the printed token to `~/.ar_claude_token`
(`chmod 600`). An `ANTHROPIC_API_KEY` or an existing `claude` login also works. A full run is 400 experiments,
about two days on one 8-GPU node, and a substantial amount of model usage; use `--budget` for a shorter run.

The repository must be committed (a run records the commit it started from; `--allow-dirty` overrides this).
The agent runs in a [bubblewrap](https://github.com/containers/bubblewrap) jail if `bwrap` works on your machine;
otherwise it runs unjailed with a warning (`--no-jail` forces this).

```bash
source env.sh
enforced/launch.sh --strategy search --batch 4 --budget 400              # Enforced
unchecked/launch.sh --arm vanilla --strategy search --batch 4 --budget 400   # Vanilla
unchecked/launch.sh --arm free --strategy search --batch 4 --budget 400      # Free

python enforced/autoresearch.py status                    # progress (unchecked/autoresearch.py for the others)
python common/localgpu.py status                          # GPU queue: running and waiting tasks
```

Each run lives in `$AR_WORK/outputs/<date>/<tag>/` (the agent's folder: `results.tsv`, `experiments/`,
`notes.md`), with its full trace in `$AR_WORK/traces/<date>/<tag>/`. Preview what the agent is told with
`python enforced/autoresearch.py instructions --strategy search` or
`python unchecked/autoresearch.py instructions --arm free --strategy search`.

## 5. Audit a finished run

```bash
python enforced/tools/audit.py <date>/<tag>        # Enforced runs
python unchecked/tools/audit.py <date>/<tag>       # Vanilla / Free runs (the arm is read from the run)
```

The auditor is a separate Claude session with read-only access to the run folder, the trace and the raw logs;
it writes `$AR_WORK/traces/<date>/<tag>/audit/audit.md` and `verdict.json` (clean / minor issues / serious issues).

## Credits and license

The model, data, tokenizer and training recipe are Karpathy's autoresearch (MIT license); the harness, checks,
jail, search loop, audits and analysis are ours. Released under the MIT license (`LICENSE`).
