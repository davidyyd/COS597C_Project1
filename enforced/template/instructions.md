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
  constants without a new mechanism), gaining capacity by making modules bigger, optimizer research
  (see "How parameters are optimized"), and exploiting the evaluation, the loss or the caps.

A clean negative result for a well-motivated idea is useful: record what you learned in `notes.md`.

## Fixed setup

- **Baseline** (`baseline_model.py`; `model.py` starts as a copy): the GPT of Karpathy's autoresearch
  (from nanochat) at the size its runs train: 8 layers, width 512, 4 heads of 128, RMSNorm without
  gains, embedding re-added before every block (x0 lambdas), RoPE then QK-norm, sliding-window
  attention (pattern SSSL), value embeddings on alternating layers, ReLU² MLP, untied output head,
  logit soft-cap 15, 50M parameters. **Baseline score: {BASELINE_VAL}** (validation bits per byte,
  lower is better).
- **Training:** ClimbMix with an 8,192-token BPE (Karpathy's autoresearch data and tokenizer),
  sequence length 2048 (every row starts with a BOS token; documents packed best-fit), 524,288 tokens
  per step, 3,815 steps (~2.0B tokens), on 8 H100s. Muon + AdamW, one shared LR fixed at {LR}, weight decay 0.1,
  5% linear warmup then linear decay to 0, bf16 mixed precision (fp32 parameters, the forward pass
  under bf16 autocast). A design must train well at this LR.
- **Score:** the final validation bits per byte on a fixed 21M-token slice of the pinned validation
  shard (reported as `val_loss` by the tools). Each experiment is trained once.

## Your research loop

{LOOP}

## Commands

Run them from this folder (or a folder inside it). They hand the work to the experiment system and
print what it returns; stopping one (Ctrl-C, a timeout) stops its work. Never end your turn while one
is running: ending your turn ends your session, which stops them and discards their results.

- **`ar-check MODEL_FILE`** (about 5 minutes; unlimited): the gate every model must pass before it
  can be trained. It checks the model on one H100 (code rules, sizes and FLOPs against the caps,
  causality, `torch.compile`, and it lists which optimizer trains each parameter), then measures
  its step time on 8 H100s over training steps 20–120. It ends with `CHECK: PASS` or
  `CHECK: FAIL (...)`. Each model is measured once: checking the same code again (comments and
  formatting do not count) returns the same result.
- **`ar-exp NAME=MODEL_FILE [NAME=MODEL_FILE ...]`** (about 12–15 minutes; at most {BATCH} per call,
  trained in parallel): trains models that passed `ar-check` with exactly the same code; anything
  else is refused. It snapshots each model file and the `idea.md` next to it, trains them, and
  prints a table of results. It blocks until all finish: give it a timeout of at least 30 minutes,
  or run it in the background and wait. Names must be new (letters, digits, `-`, `_`).

Every experiment needs an `idea.md` next to its model file: a title on the first line, then the
hypothesis, the change you made (ideally as a formula or a few lines), what result would confirm or
refute it, and the closest known work and how your idea differs.

## Results

Each experiment gets `experiments/NAME/` with `model.py` and `idea.md` (exactly what was trained),
`check.txt` (its `ar-check` report), `train.log` (training and validation loss over time),
`result.json`, and `error.txt` after a crash (the traceback lines in your model file). Its status is
`trained` (with a `val_loss`), `diverged` (the loss became NaN or infinite) or `crash-train`.

`results.tsv` has one row per experiment: `name status val_loss step_ms train_step_ms total_params
flops_ratio finished idea` (`step_ms` from `ar-check`; `train_step_ms` measured during training,
recorded only). Its first row is the baseline. `ar-exp` commits every experiment to git.

## Model interface

A model file must provide:

- `build_model(vocab_size) -> nn.Module`;
- `model(inputs)`: `inputs` int32 `[B, T]` → logits `[B, T, vocab_size]` (the loss is fixed: summed
  token cross-entropy on these logits);
- `model.init_weights()`: initializes every parameter (initialization is part of the architecture);
- `model.param_components()`: maps every parameter name to `attention`, `mlp`, `embedding`, `head`
  or `other`, honestly by its role (`nn.Embedding` weights are `embedding`, other vocab-sized
  tensors `embedding` or `head`). The component sets both its size cap and how it is optimized.

The model must stay a causal language model (logits at position t depend only on tokens 0..t of the
same sequence), every parameter must receive a gradient, and it must work under `torch.compile`. A
model file may import only `torch` and standard helpers (`math`, `functools`, `typing`, ...), and
may not read or write files, load weights or data, open connections, run processes, or touch
gradients (hooks, `.grad`, `requires_grad`). Every model trains from scratch.

Precision belongs to the training system, like the optimizer. Parameters and buffers are fp32, and
the model's code is dtype-agnostic: it computes in fp32 and the harness's bf16 autocast alone decides
which ops run in bf16. A model may not lower precision itself: no bf16, fp16, fp8 or int8 tensors or
casts (`.bfloat16()`, `.half()`, `dtype=torch.bfloat16`), no casts that follow another tensor's dtype
(`.type_as`, `.to(x.dtype)`, `.to(other)`), no `.dtype` checks, no autocast or TF32 settings, no
custom kernels or ops, and no `torch.compile` or compiler settings of its own. Upcasts (`.float()`,
`.to(torch.float32)`) are fine. Speed must come from the architecture, not from numerics. `ar-check`
checks this from the source and by running the model with autocast off, where it must compute
entirely in fp32.

## How parameters are optimized

The training system assigns each parameter's optimizer from its component and shape:

| parameter | optimizer |
|---|---|
| 2-D, component `attention` or `mlp` | Muon on the whole matrix |
| 3-D or more, `attention` or `mlp`, shape `[..., rows, cols]` | Muon on each `[rows, cols]` matrix separately (e.g. one head) |
| everything else (1-D gains and scalars, `embedding`, `head`, `other`) | AdamW (no weight decay on 1-D parameters) |

Store each weight in the shape of its structure: a projection producing independent heads is
`[heads, head_dim, in_dim]` (the baseline keeps Karpathy's separate 2-D q, k and v projections, `[512, 512]` each), a map that mixes all
its inputs is 2-D. Muon moves every entry of a matrix by about 0.2 × LR per step whatever its size,
so a small matrix feeding gates changes its outputs quickly. Reshaping or relabelling weights,
weight multipliers or re-parameterizations whose only purpose is a different effective step size
are optimizer research, not architecture, and are out of scope.

## Constraints

`ar-check` enforces these limits. A candidate may cost at most 5% more than the baseline:

1. **Step time** on 8 H100s ≤ {STEP_CAP} ms (baseline {STEP_BASE} ms; repeated measurements of the
   baseline vary by about ±1 ms).
2. **FLOPs per token** ≤ {FLOPS_CAP} × baseline.
3. **Parameters per component** ≤ {COMPONENT_CAP} × baseline:

{CAPS_TABLE}
{REVIEW}
## Rules

- Your shell sees only this folder; this machine has no GPU and no cluster access for you. All GPU
  work goes through `ar-check` and `ar-exp`.
- Write only model files, `idea.md` files, `notes.md`, and folders of your own (e.g. `cands/`).
  `experiments/`, `results.tsv`, `baseline_model.py`, this file and the git history are read-only.
- You may use shell commands, Python (CPU) and the internet. The software environment is fixed: do
  not install packages.
- `notes.md` is your memory: this session is long and older parts get compacted, so keep in it what
  you learned, the best models so far, and what to try next.
- **Budget: {BUDGET} experiments** (`ar-exp`; every call reports how many are left). `ar-check` is
  unlimited. When the budget is used up, bring `notes.md` up to date (what you learned, the best
  models and why) and end your turn: the run ends there. Until then, never stop: do not ask whether
  to continue, and do not end your turn.
