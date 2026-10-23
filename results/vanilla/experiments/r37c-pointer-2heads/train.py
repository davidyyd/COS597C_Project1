"""
Autoresearch training script: one file with the model, the optimizer and the training loop.
Runs on 8 GPUs (torchrun --nproc_per_node=8 train.py) for NUM_STEPS optimizer steps of 2**19 tokens.

The model (Karpathy's autoresearch GPT, from nanochat), the optimizer (Muon for 2-D attention/MLP matrices,
AdamW for everything else, one LR, weight decay 0.1, 5% warmup then linear decay to 0), the loss (summed
token cross-entropy under bf16 autocast) and the data order are the same as in the karpathy-arch harness,
so the baseline score is the same.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.optim import AdamW

from prepare import MAX_SEQ_LEN, NUM_STEPS, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """True if the layer has a value embedding (alternating, the last one always included)"""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # value residual (ResFormer): mix in the value embedding with an input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve.to(v.dtype)     # bf16 value path
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if window is None:          # full causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                       # sliding window: keys t-window..t (their FA3 window_size=(window, 0))
            y = flex_attention(q, k, v, block_mask=window)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window):
        x = x + self.attn(norm(x), ve, cos_sin, window)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.windows = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # pointer-sentinel copy head: p(y) = a_sent * p_vocab(y) + sum_{s<t} a_ts [token_{s+1} == y]
        self.ptr_dim = 64
        self.ptr_heads = 2
        self.ptr_q = nn.Linear(config.n_embd, self.ptr_heads * self.ptr_dim, bias=False)
        self.ptr_k = nn.Linear(config.n_embd, self.ptr_heads * self.ptr_dim, bias=False)
        self.ptr_sentinel = nn.Linear(config.n_embd, 1, bias=False)
        self.ptr_gain = nn.Parameter(torch.ones(self.ptr_heads * self.ptr_dim))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._masks = {}            # (window, T, device) -> flex_attention BlockMask

    @torch.no_grad()
    def init_weights(self):
        s_ptr = 3**0.5 * self.config.n_embd**-0.5
        torch.nn.init.uniform_(self.ptr_q.weight, -s_ptr, s_ptr)
        torch.nn.init.uniform_(self.ptr_k.weight, -s_ptr, s_ptr)
        torch.nn.init.zeros_(self.ptr_sentinel.weight)
        self.ptr_gain.fill_(1.0)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=self.lm_head.weight.device)
        self.cos, self.sin = cos, sin
        for w in set(self.windows):     # built here, outside torch.compile, and reused by every forward
            if w is not None:
                self._mask(w, self.config.sequence_len, self.lm_head.weight.device)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()   # bf16 tables, as in the original nanochat model:
        # keeps q, k in bf16 so the flex-attention layers run in bf16 like the SDPA layers do under autocast
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        """per layer: None for full attention (L), else the sliding window (S: half the context); last layer full"""
        long_window, short_window = config.sequence_len, config.sequence_len // 2
        windows = [short_window if config.window_pattern.upper()[i % len(config.window_pattern)] == "S" else None
                   for i in range(config.n_layer)]
        windows[-1] = None
        return [None if w is not None and w >= long_window else w for w in windows]

    def _pointer_logprob(self, x, idx, targets, lp_vocab):
        """log p(y_t) of the pointer-sentinel mixture, computed densely over causal query chunks
        (queries in chunk c only score keys before the chunk's end) with finite masking:
        lse_all = log sum_{s<t} e^{S_ts}, lse_match = log sum_{s<t, token_{s+1} = y_t} e^{S_ts},
        L = logaddexp(lse_all, z_t);  p = e^{z-L} p_vocab(y) + e^{lse_match - L}."""
        B, T, C = x.shape
        P = self.ptr_heads
        qc = (norm(self.ptr_q(x).view(B, T, P, -1)) * self.ptr_gain.to(x.dtype).view(P, -1)).transpose(1, 2)   # (B,P,T,d)
        kc = norm(self.ptr_k(x).view(B, T, P, -1)).transpose(1, 2)
        z = self.ptr_sentinel(x).float().squeeze(-1)                                    # (B, T)
        nxt = torch.cat([idx[:, 1:], idx[:, -1:]], dim=1)                              # token s+1 at s
        t = torch.arange(T, device=x.device)
        lse_all, lse_match = [], []
        n_chunks = 4
        Tc = T // n_chunks
        for c in range(n_chunks):
            q0, q1 = c * Tc, (c + 1) * Tc                                               # keys 0 .. q1-1
            S = (qc[:, :, q0:q1] @ kc[:, :, :q1].transpose(-1, -2)).float() * self.ptr_dim ** -0.5   # (B,P,Tq,Tk)
            earlier = (t[q0:q1].view(1, -1, 1) > t[:q1].view(1, 1, -1)).unsqueeze(1)    # s < t
            match = earlier & (nxt[:, :q1].unsqueeze(1) == targets[:, q0:q1].unsqueeze(2)).unsqueeze(1)
            # joint normalisation over both heads' keys: logsumexp over (head, key)
            lse_all.append(torch.logsumexp(S.masked_fill(~earlier, -1e4).transpose(1, 2).flatten(2), dim=-1))
            lse_match.append(torch.logsumexp(S.masked_fill(~match, -1e4).transpose(1, 2).flatten(2), dim=-1))
        lse_all, lse_match = torch.cat(lse_all, dim=1), torch.cat(lse_match, dim=1)
        L = torch.logaddexp(lse_all, z)
        return torch.logaddexp(z - L + lp_vocab, lse_match - L)

    def _ptr_bias(self, idx, q0, q1):
        return 0.0

    def _mask(self, window, T, device):
        """flex_attention BlockMask of the sliding window (built once, outside torch.compile)"""
        key = (window, T, str(device))
        if key not in self._masks:
            def sliding(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= window)
            self._masks[key] = create_block_mask(sliding, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device)
        return self._masks[key]

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx)).bfloat16()     # bf16 residual stream (like every other activation)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            w = self.windows[i]
            x = block(x, ve, cos_sin,
                      None if w is None else self._mask(w, T, idx.device))
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            lp_vocab = -F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none").view(B, T)
            nll = -self._pointer_logprob(x, idx, targets, lp_vocab)
            if reduction == "none":
                return nll.view(-1)
            return nll.sum() if reduction == "sum" else nll.mean()
        return logits       # (vocab part only; the copy mixture needs the target, see _pointer_logprob)

    def param_components(self):
        # component of every parameter: for the per-component size caps, and (with its shape) for the
        # harness's choice of optimizer
        components = {}
        for name, p in self.named_parameters():
            if name.startswith(("transformer.wte.", "value_embeds.")):
                components[name] = "embedding"
            elif name.startswith("lm_head."):
                components[name] = "head"
            elif ".attn." in name and p.ndim >= 2:
                components[name] = "attention"
            elif ".mlp." in name and p.ndim >= 2:
                components[name] = "mlp"
            else:
                components[name] = "other"
        return components


# ---------------------------------------------------------------------------
# Optimizer (Muon for 2-D attention/MLP matrices, AdamW for the rest) and LR schedule
# ---------------------------------------------------------------------------

WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
MUON_MU = 0.95
WARMUP_FRAC = 0.05                    # linear warmup over the first 5% of steps, then linear decay to 0

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

def lr_multiplier(step: int, train_steps: int = NUM_STEPS):
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

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LR = 1e-2                 # one learning rate for every parameter group
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens per optimizer step
DEVICE_BATCH_SIZE = 32    # sequences per GPU per forward/backward (8 GPUs x 32 x 2048 = 2**19)
SEED = 1337

# ---------------------------------------------------------------------------
# Setup: distributed, tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
rank, world_size = dist.get_rank(), dist.get_world_size()
master = rank == 0
if not master:   # only rank 0 prints
    import builtins
    builtins.print = lambda *a, **k: None
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

torch.manual_seed(SEED)
model = GPT(GPTConfig(vocab_size=vocab_size)).to(device)
model.init_weights()
for p in model.parameters():
    dist.broadcast(p.detach(), 0)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

optimizers = build_optimizers(model, LR)
model = torch.compile(model, dynamic=False)

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * world_size
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
print(f"Training steps: {NUM_STEPS}, gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
for step in range(NUM_STEPS):
    torch.cuda.synchronize()
    t0 = time.time()
    train_loss = torch.zeros((), device=device)
    for micro_step in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        with autocast_ctx:
            loss = model(x, y, reduction="sum")      # summed token cross-entropy
        train_loss += loss.detach()
        loss.backward()
    dist.all_reduce(train_loss)
    train_loss_f = train_loss.item() / TOTAL_BATCH_SIZE
    for p in model.parameters():                   # sum gradients across GPUs
        dist.all_reduce(p.grad)
    set_hparams(optimizers, step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 10:
        total_training_time += dt
    if step % 100 == 0:
        print(f"step {step:05d} ({100 * step / NUM_STEPS:.1f}%) | loss: {train_loss_f:.6f} | lrm: {lr_multiplier(step):.2f} | dt: {dt*1000:.0f}ms", flush=True)

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {NUM_STEPS * TOTAL_BATCH_SIZE / 1e6:.1f}")
print(f"num_steps:        {NUM_STEPS}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
dist.destroy_process_group()
