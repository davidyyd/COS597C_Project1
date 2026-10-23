# Language-model training research

## Mission

Your goal is to get the lowest possible `val_bpb` from `train.py`. You may change anything in
`train.py`.

- **Read the literature.** You can search the web and read any paper or code.

## Setup

- **Baseline** (`baseline_train.py`; `train.py` starts as a copy): one training script with the model (the GPT of Karpathy's
  autoresearch, from nanochat), the optimizer and the training loop. The model: 8 layers,
  width 512, 4 heads of 128, RMSNorm without gains, embedding re-added before every block (x0 lambdas),
  RoPE then QK-norm, sliding-window attention (pattern SSSL), value embeddings on alternating layers,
  ReLU² MLP, untied output head, logit soft-cap 15, 50M parameters. **Baseline score: 0.958950**
  (validation bits per byte, lower is better).
- **Training** (the setup in `train.py` and `prepare.py`): ClimbMix with an 8,192-token BPE, sequence
  length 2048 (every row starts with a BOS token; documents packed best-fit), 524,288 tokens per step,
  3,815 steps (`NUM_STEPS` in `prepare.py`, ~2.0B tokens), on 8 H100s (`torchrun`). Muon for the 2-D
  attention and MLP matrices and AdamW for everything else, one learning rate (1e-2) for all of them,
  weight decay 0.1, 5% linear warmup then linear decay to 0, summed token cross-entropy under bf16
  autocast with fp32 parameters, seed 1337.
- **Score:** the `val_bpb` that `train.py` prints at the end (the baseline gets it from `evaluate_bpb`
  in `prepare.py`: validation bits per byte on a fixed 21M-token slice of the pinned validation shard).
  Each experiment is trained once.

## Your research loop

This run uses **search over a population of training scripts**, in rounds of 4 experiments trained in
parallel. There is no single current script: every experiment you have run is a possible starting
point, and you choose what to build on. Your score is the best val_bpb of any experiment.

Repeat, one round at a time:

1. Read `notes.md`, `results.tsv`, and whatever you need in `experiments/`. Every past experiment's
   `train.py` is in `experiments/<name>/train.py`; the baseline is `baseline_train.py`.
2. Design 4 experiments for this round. Each starts from a `train.py` you choose (the baseline or
   any earlier experiment) and changes it. Balance two kinds:
   - **exploit:** build on the strongest results so far (refine one, or combine ideas that worked
     in different experiments);
   - **explore:** a new direction, usually starting from the baseline, clearly different from
     everything tried so far.
   Explore more in early rounds and exploit more as strong results emerge, but always keep at
   least one explore experiment per round.
3. For each, create `cands/<name>/` with its `train.py` and `idea.md`. In `idea.md`, after the
   title, state the script it starts from (`Parent: <experiment name or baseline>`) and whether it
   is exploit or explore.
4. Run them together: `ar-exp <name1>=cands/<name1>/train.py <name2>=cands/<name2>/train.py ...`
   (new descriptive names each round, e.g. `r05a-routing-on-r04b`).
5. Update `notes.md`: the round's results, what you learned, the best results so far (names and
   scores), and the plan for the next round.

Use all 4 slots every round. Crashed and diverged experiments count toward the budget.

## Command

Run it from this folder (or a folder inside it). It hands the work to the experiment system and prints
what it returns; stopping it (Ctrl-C, a timeout) stops its work. Never end your turn while it is running:
ending your turn ends your session, which stops it and discards its results.

- **`ar-exp NAME=TRAIN_FILE [NAME=TRAIN_FILE ...]`** (about 10–15 minutes; at most 4 per call, run
  in parallel): runs each `train.py` on 8 H100s (`torchrun`, next to `prepare.py`) with a limit
  of 60 minutes, snapshots it and the `idea.md` next to it, and prints a table of results. It blocks
  until all finish: give it a timeout of at least 75 minutes, or run it in the background and wait.
  Names must be new (letters, digits, `-`, `_`).

Every experiment needs an `idea.md` next to its `train.py`: a title on the first line, then what you
changed and why.

## Results

Each experiment gets `experiments/NAME/` with `train.py` and `idea.md` (exactly what was run), `run.log`
(the last 400 lines of its output) and `result.json`. Its status is `trained` (it printed a `val_bpb`),
`crash` (it did not; see `run.log`) or `timeout`.

`results.tsv` has one row per experiment: `name status val_bpb training_seconds peak_vram_mb num_steps
num_params_M finished idea` (from the summary `train.py` prints). Its first row is the baseline. `ar-exp`
commits every experiment to git.

## Environment

- Your shell sees only this folder; this machine has no GPU and no cluster access for you. All GPU
  work goes through `ar-exp`.
- `experiments/`, `results.tsv`, `baseline_train.py`, `prepare.py`, this file and the git history are
  read-only; `ar-exp` runs your `train.py` next to an unmodified `prepare.py`.
- You may use shell commands, Python (CPU) and the internet. The software environment is fixed: no
  package installs.
- `notes.md` is your memory: this session is long and older parts get compacted, so keep in it what
  you learned, the best models so far, and what to try next.
- **Budget: 400 experiments** (`ar-exp`; every call reports how many are left). When the budget is
  used up, bring `notes.md` up to date (what you learned, the best results and why) and end your turn:
  the run ends there. Until then, never stop: do not ask whether to continue, and do not end your turn.
