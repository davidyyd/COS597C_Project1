# Language-model architecture research

## Mission

You do language-model architecture research in a controlled testbed. The data, optimizer, learning
rate, schedule, token budget and evaluation are fixed; only the model definition changes, so
differences in validation loss come from the architecture.

Your goal is to find **new architectural ideas** that make the model better: better inductive
biases, information flow, or use of capacity, in ways that would plausibly carry over to other scales.
Treat each experiment as a test of a clear hypothesis about a weakness of the current model.

- **Read the literature.** You can search the web and read any paper or code. Use it to understand
  the problem, to find starting points, and to check what is already known.
- **Contribute something new.** Do not simply re-implement a known improvement (anything published,
  or in public speedrun / open-source code). A known method can be your starting point, but the
  experiment should test an idea of your own.
- **Prefer simple ideas.** The best ideas fit in a short formula or a few lines of code. Favour one
  clean, principled change over an elaborate heuristic with many hand-set parts, special cases,
  thresholds or magic constants. If an idea needs a lot of machinery to work, look for the simple
  principle underneath it.
- **Out of scope:** hyperparameter tuning presented as architecture (widths, depths, head counts,
  constants without a new mechanism), gaining capacity by making modules bigger, optimizer research, and exploiting the evaluation, the loss or the caps.

A clean negative result for a well-motivated idea is useful: record what you learned in `notes.md`.

## Fixed setup

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
  autocast with fp32 parameters, seed 1337. This training setup (optimizer, learning rate, schedule,
  batch size, number of steps, precision, data) is fixed: change only the model definition.
- **Score:** the `val_bpb` that `train.py` prints at the end: `evaluate_bpb` in `prepare.py`, the final
  validation bits per byte on a fixed 21M-token slice of the pinned validation shard. Each experiment is
  trained once.

## Your research loop

This run uses **search over a population of models**, in rounds of 4 experiments trained in
parallel. There is no single current model: every experiment you have run is a possible starting
point, and you choose what to build on. Your score is the best val_loss of any experiment.

Repeat, one round at a time:

1. Read `notes.md`, `results.tsv`, and whatever you need in `experiments/`. Every past experiment's
   `train.py` is in `experiments/<name>/train.py`; the baseline is `baseline_train.py`.
2. Design 4 experiments for this round. Each starts from a `train.py` you choose (the baseline or
   any earlier experiment) and makes one change. Balance two kinds:
   - **exploit:** build on the strongest models so far (refine one, or combine ideas that worked
     in different experiments);
   - **explore:** a new direction, usually starting from the baseline, clearly different from
     everything tried so far.
   Explore more in early rounds and exploit more as strong models emerge, but always keep at
   least one explore experiment per round.
3. For each, create `cands/<name>/` with its `train.py` and `idea.md`. In `idea.md`, after the
   title, state the model it starts from (`Parent: <experiment name or baseline>`) and whether it
   is exploit or explore.
4. Run them together: `ar-exp <name1>=cands/<name1>/train.py <name2>=cands/<name2>/train.py ...`
   (new descriptive names each round, e.g. `r05a-routing-on-r04b`).
5. Update `notes.md`: the round's results, what you learned, the best models so far (names and
   scores), and the plan for the next round.

Use all 4 slots every round. Crashed and diverged experiments count toward the budget.

## Command

Run it from this folder (or a folder inside it). It hands the work to the experiment system and prints
what it returns; stopping it (Ctrl-C, a timeout) stops its work. Never end your turn while it is running:
ending your turn ends your session, which stops it and discards its results.

- **`ar-exp NAME=TRAIN_FILE [NAME=TRAIN_FILE ...]`** (about 10–15 minutes; at most 4 per call, run
  in parallel): runs each `train.py` on 8 H100s (`torchrun`, next to the fixed `prepare.py`) with a limit
  of 60 minutes, snapshots it and the `idea.md` next to it, and prints a table of results. It blocks
  until all finish: give it a timeout of at least 75 minutes, or run it in the background and wait.
  Names must be new (letters, digits, `-`, `_`).

Every experiment needs an `idea.md` next to its `train.py`: a title on the first line, then the
hypothesis, the change you made (ideally as a formula or a few lines), what result would confirm or
refute it, and the closest known work and how your idea differs.

## Results

Each experiment gets `experiments/NAME/` with `train.py` and `idea.md` (exactly what was run), `run.log`
(the last 400 lines of its output) and `result.json`. Its status is `trained` (it printed a `val_bpb`),
`crash` (it did not; see `run.log`) or `timeout`.

`results.tsv` has one row per experiment: `name status val_bpb training_seconds peak_vram_mb num_steps
num_params_M finished idea` (from the summary `train.py` prints). Its first row is the baseline. `ar-exp`
commits every experiment to git.

## Constraints

A candidate may cost at most 5% more than the baseline in training time (`training_seconds`), in FLOPs
per token and in parameters. The model must stay a causal language model (the prediction at position t
depends only on tokens 0..t of the same sequence) and train from scratch.

## Rules

- Your shell sees only this folder; this machine has no GPU and no cluster access for you. All GPU
  work goes through `ar-exp`.
- Write only `train.py` files, `idea.md` files, `notes.md`, and folders of your own (e.g. `cands/`).
  `experiments/`, `results.tsv`, `baseline_train.py`, `prepare.py`, this file and the git history are
  read-only.
- You may use shell commands, Python (CPU) and the internet. The software environment is fixed: do
  not install packages.
- `notes.md` is your memory: this session is long and older parts get compacted, so keep in it what
  you learned, the best models so far, and what to try next.
- **Budget: 400 experiments** (`ar-exp`; every call reports how many are left). When the budget is
  used up, bring `notes.md` up to date (what you learned, the best models and why) and end your turn:
  the run ends there. Until then, never stop: do not ask whether to continue, and do not end your turn.
