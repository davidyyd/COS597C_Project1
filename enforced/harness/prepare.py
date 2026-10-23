"""
prepare.py -- FROZEN. Data, loss, optimizer, schedule, and measurement utilities.

Training recipe (data, batch, schedule, optimizer family from modded-nanogpt
records/track_3_optimization/2026-08-28/lr_sweep_2b_warmup0p4pct_muon_adamw; LR re-swept for the
current baseline on 2026-10-07, see baseline/baseline.json):
  * branch karpathy-arch: the data, tokenizer, context and metric of Karpathy's autoresearch
    (github.com/karpathy/autoresearch @ 228791f): ClimbMix shards, their 8192-token BPE, seq_len 2048,
    rows packed BOS-aligned best-fit (data/prepare_climbmix.py writes them once), 524,288 tokens/step,
    3815 steps (~2.0B tokens)
  * Muon (non-Nesterov, mu=0.95, RMS-matched to AdamW) + AdamW(0.9, 0.95); which parameter gets which
    is decided here from its shape and component (split_params), never by the model file:
    2-D attention/mlp -> Muon on the whole matrix, 3-D+ attention/mlp [..., rows, cols] -> Muon on each
    [rows, cols] slice (e.g. one attention head), everything else -> AdamW
  * one shared base LR (fixed at 1e-2, the baseline's best on this branch), weight decay 0.1 on
    every parameter with 2+ dimensions (none on 1-D gains and scalars, as GPT-2 / nanoGPT),
    5% linear warmup, then linear decay to 0 (changed 2026-10-07 from 0.4% warmup + 70% cooldown)
  * mixed precision: fp32 parameters, forward under bf16 autocast (LossWrapper)
  * validation: bits per byte (their evaluate_bpb) on their pinned val shard, 40 * 524,288 tokens; it is
    reported under the name val_loss everywhere in this harness
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AR_DATA") or ROOT.parent / "data" / "climbmix")   # see the repository's env.sh
TRAIN_PATTERN = "train_rows.bin"
VAL_PATTERN = "val_rows.bin"
TOKEN_BYTES = DATA_DIR / "tokenizer" / "token_bytes.npy"   # UTF-8 bytes of each token (0: special tokens)

VOCAB_SIZE = 8192
SEQ_LEN = 2048
BATCH_SIZE = 8 * 64 * 1024            # tokens per optimizer step
TRAIN_STEPS = 3815                    # 3815 * 524288 = 2,000,158,720 tokens
VAL_TOKENS = 40 * 524288
MAX_MICROBATCH = 128                  # sequences per forward/backward
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
MUON_MU = 0.95
WARMUP_FRAC = 0.05                    # linear warmup over the first 5% of steps, then linear decay to 0
LR = "1e-2"        # the one LR every experiment trains at (the baseline's best; baseline/baseline.json per_lr)

# fairness caps (baseline values are recorded in harness/caps.json)
# fixed cost: a candidate may exceed the baseline by at most 5% in step time, FLOPs per token and
# each component's parameter count (values in harness/caps.json, measured by tools/measure_baseline.py)
STEP_TIME_RATIO_CAP = 1.05            # checked by the speed benchmark (ar-check / the gate of ar-exp), steps 20..120
FLOPS_RATIO_CAP = 1.05
STEP_CHECK_WINDOW = (20, 120)
COMPONENTS = ("attention", "mlp", "embedding", "head", "other")
COMPONENT_RATIO_CAP = 1.05            # each component <= 1.05x its baseline parameter count
OTHER_ABS_CAP = 1_000_000             # "other" (norms, gates, scalars) gets a flat allowance instead (~0.6% of the model)


########################################
#              Dataloader              #
########################################

def _load_rows(file: Path):
    """packed rows written by data/prepare_climbmix.py: header (256 int32), then [rows, SEQ_LEN + 1] uint16"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20261009 and header[1] == 2, "not a packed-rows file (data/prepare_climbmix.py)"
    nrows, row = int(header[2]), int(header[3])
    assert row == SEQ_LEN + 1, f"rows of {row} tokens, expected {SEQ_LEN + 1}"
    import numpy as np
    return np.memmap(file, dtype=np.uint16, mode="r", offset=256 * 4, shape=(nrows, row))

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=SEQ_LEN):
    """batch_size tokens per step = batch_size // seq_len rows, split evenly over the ranks, in file order.
    Every row starts with BOS (their best-fit packing); inputs are row[:-1], targets row[1:]."""
    rows = _load_rows(DATA_DIR / filename_pattern)
    assert batch_size % (seq_len * dist.get_world_size()) == 0
    per_step = batch_size // seq_len
    local = per_step // dist.get_world_size()
    # this rank's rows of every step, read into memory once: a step never waits on the shared filesystem
    # (read per step from the memmap, a step's time depended on whether the node's page cache already held
    # those rows: ~66 ms warm vs ~70 ms cold)
    import numpy as np
    n_steps = len(rows) // per_step
    starts = np.arange(n_steps) * per_step + dist.get_rank() * local
    mine = np.ascontiguousarray(rows[(starts[:, None] + np.arange(local)[None, :]).reshape(-1)])
    mine = torch.from_numpy(mine.reshape(n_steps, local, -1).astype(np.int32))   # token ids < 65536
    step = 0
    while True:
        assert step < n_steps, "out of packed rows (data/prepare_climbmix.py writes exactly enough)"
        buf = mine[step].pin_memory()
        inputs = buf[:, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[:, 1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        step += 1
        yield inputs, targets

def token_bytes(device="cuda"):
    import numpy as np
    return torch.from_numpy(np.load(TOKEN_BYTES)).to(device)


########################################
#            Model + frozen loss       #
########################################

def load_model_module(model_file):
    # torch.compile resolves the module by name, so each model file loaded in this process keeps its own
    # name (the first is "ar_model"); reusing one name would break models loaded earlier
    name = "ar_model" if "ar_model" not in sys.modules else f"ar_model_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(name, str(model_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class LossWrapper(nn.Module):
    """Frozen loss: summed token cross-entropy on the model's logits. Precision is the harness's:
    parameters stay fp32 and the forward pass runs under bf16 autocast (standard mixed precision)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor, targets: Tensor, weights: Tensor = None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=inputs.is_cuda):
            logits = self.model(inputs)
        logits = logits.float()
        if weights is None:
            return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
        # validation: nats summed over the targets with weight 1 (bits per byte skips 0-byte special tokens)
        nats = F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="none")
        return (nats * weights.view(-1)).sum()

def build(model_file, device="cuda"):
    """Build model + loss wrapper from a model.py file. Call torch.manual_seed first."""
    module = load_model_module(model_file)
    model = module.build_model(VOCAB_SIZE)
    assert callable(getattr(model, "init_weights", None)), "model.init_weights() is required"
    assert callable(getattr(model, "param_components", None)), "model.param_components() is required"
    model = model.to(device)
    return model, LossWrapper(model)


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # tuned five-step Newton-Schulz coefficient schedule
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95):
    momentum.lerp_(grad, 1 - mu)
    update = momentum
    update = zeropower_via_newtonschulz5(update)
    # Moonshot/Kimi RMS-matching rule: match the per-parameter RMS of an AdamW update
    update *= 0.2 * max(grad.size(-2), grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    """2-D parameters are orthogonalized as one matrix; a parameter with more dimensions
    [..., rows, cols] is a stack of independent matrices (heads, experts, groups), each
    orthogonalized on its own."""
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        assert all(p.ndim >= 2 for p in params)
        params = sorted(params, key=lambda x: x.numel(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    if p.ndim == 2:
                        update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    else:
                        stack = (-1, p.size(-2), p.size(-1))
                        update = muon_update(p.grad.reshape(stack), state["momentum"].view(stack),
                                             mu=group["mu"]).view_as(p)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

MUON_COMPONENTS = ("attention", "mlp")

def optimizer_of(component: str, shape) -> str:
    """The fixed rule: 'muon' (2-D attention/mlp), 'muon per slice' (3-D+ attention/mlp), or 'adamw'."""
    if component in MUON_COMPONENTS and len(shape) == 2:
        return "muon"
    if component in MUON_COMPONENTS and len(shape) > 2:
        return "muon per slice"
    return "adamw"

def split_params(model: nn.Module):
    """Optimizer of every parameter, from its component (model.param_components()) and shape.
    Returns (muon params, adamw params, {param name: 'muon' | 'muon per slice' | 'adamw'})."""
    mapping = model.param_components()
    plan = {name: optimizer_of(mapping[name], p.shape) for name, p in model.named_parameters()}
    named = dict(model.named_parameters())
    muon_params = [named[n] for n, o in plan.items() if o.startswith("muon")]
    adamw_params = [named[n] for n, o in plan.items() if o == "adamw"]
    return muon_params, adamw_params, plan

def build_optimizers(model: nn.Module, base_lr: float):
    muon_params, adamw_params, _ = split_params(model)
    adamw_groups = [dict(params=[p for p in adamw_params if p.ndim >= 2], weight_decay=WEIGHT_DECAY),
                    dict(params=[p for p in adamw_params if p.ndim < 2], weight_decay=0.0)]
    optimizers = [
        AdamW([g for g in adamw_groups if g["params"]], lr=base_lr, betas=ADAM_BETAS, eps=1e-10, fused=True),
        Muon(muon_params, lr=base_lr, weight_decay=WEIGHT_DECAY, mu=MUON_MU),
    ]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers

def lr_multiplier(step: int, train_steps: int = TRAIN_STEPS):
    """linear warmup to 1 over the first WARMUP_FRAC of the steps (191 of 3815), then linear decay to 0"""
    warmup_steps = round(WARMUP_FRAC * train_steps)
    assert 0 <= step < train_steps
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return (train_steps - step) / (train_steps - warmup_steps)

def set_hparams(optimizers, step: int):
    eta = lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#            Measurement               #
########################################

def count_params(model: nn.Module):
    """total and non-embedding params. Non-embedding excludes nn.Embedding weights and any
    parameter with a vocab-sized dimension (the output head)."""
    embed_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    total = sum(p.numel() for p in model.parameters())
    non_embedding = sum(p.numel() for p in model.parameters()
                        if id(p) not in embed_ids and VOCAB_SIZE not in p.shape)
    return {"total": total, "non_embedding": non_embedding}

def component_params(model: nn.Module):
    """Parameter count per component, from model.param_components() {param name: component}.
    nn.Embedding weights must be 'embedding'; other vocab-sized tensors 'embedding' or 'head';
    every remaining tensor 'attention', 'mlp', or 'other'."""
    assert callable(getattr(model, "param_components", None)), "model.param_components() is required"
    mapping = model.param_components()
    named = dict(model.named_parameters())
    missing, unknown = set(named) - set(mapping), set(mapping) - set(named)
    assert not missing, f"param_components() is missing parameters: {sorted(missing)[:10]}"
    assert not unknown, f"param_components() names unknown parameters: {sorted(unknown)[:10]}"
    embed_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    counts = {c: 0 for c in COMPONENTS}
    for name, p in named.items():
        comp = mapping[name]
        assert comp in COMPONENTS, f"{name}: unknown component {comp!r} (use one of {COMPONENTS})"
        if id(p) in embed_ids:
            assert comp == "embedding", f"{name} is an nn.Embedding weight and must be 'embedding'"
        elif VOCAB_SIZE in p.shape:
            assert comp in ("embedding", "head"), f"{name} is vocab-sized and must be 'embedding' or 'head'"
        else:
            assert comp in ("attention", "mlp", "other"), f"{name} is not vocab-sized and must be 'attention', 'mlp' or 'other'"
        counts[comp] += p.numel()
    return counts

def component_caps(baseline_counts):
    caps = {c: int(COMPONENT_RATIO_CAP * n) for c, n in baseline_counts.items()}
    caps["other"] = max(caps["other"], OTHER_ABS_CAP)
    return caps

def flops_per_token(wrapper: nn.Module, device, batch=2):
    """forward+backward FLOPs per token, counted by torch's FlopCounterMode (uncompiled)."""
    from torch.utils.flop_counter import FlopCounterMode
    inputs = torch.randint(0, VOCAB_SIZE, (batch, SEQ_LEN), device=device, dtype=torch.int32)
    targets = torch.randint(0, VOCAB_SIZE, (batch, SEQ_LEN), device=device, dtype=torch.int64)
    counter = FlopCounterMode(display=False)
    with counter:
        wrapper(inputs, targets).backward()
    wrapper.zero_grad(set_to_none=True)
    return counter.get_total_flops() / (batch * SEQ_LEN)

def is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
