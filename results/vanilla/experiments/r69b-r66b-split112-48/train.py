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


CONV_LAYERS = (0,)          # layers whose attention is replaced by a causal depthwise conv over the values
LOCAL_WINDOWS = {1: 128}    # early layers: short local attention windows


LOCAL_HEADS = 2       # in sliding-window layers, the first LOCAL_HEADS heads use a short local window
LOCAL_HEAD_WINDOW = 128


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
        self.conv_k = 4
        self.conv = nn.Parameter(torch.zeros(self.conv_k * self.n_embd)) if layer_idx in CONV_LAYERS else None   # 1-D: AdamW
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False) if layer_idx not in CONV_LAYERS else None
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) if layer_idx not in CONV_LAYERS else None
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # selective smear: each token gates (per head) how much of its predecessor its query / key absorbs
        self.smear_gate = nn.Linear(self.ve_gate_channels, 2 * self.n_head, bias=False) if layer_idx not in CONV_LAYERS else None
        # this layer's small extra parameters live in packed GPT-level tensors (one all-reduce each):
        # out_gate (n_head x 32): per-head output gate read from the query token (lets a head write nothing)
        # scalars: [q_gain-1 (n_head x head_dim) | key_smear (kv_dim) | query_smear (n_embd)]
        #   log_temp: per-head log-temperature on the QK-normed logits (QK-norm caps |logit| at sqrt(head_dim))
        #   key_smear: per-channel weight of the previous position's key, init 0

    def forward(self, x, ve, cos_sin, window, out_gate, scalars):
        B, T, C = x.size()
        if self.conv is not None:   # conv layer: causal depthwise conv over the values instead of attention
            v = self.c_v(x)
            if ve is not None:
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = (v.view(B, T, self.n_kv_head, self.head_dim)
                     + gate.unsqueeze(-1) * ve.view(B, T, self.n_kv_head, self.head_dim)).view(B, T, C)
            w = self.conv.view(self.conv_k, C).to(v.dtype)
            y = sum(w[j] * F.pad(v[:, :T - j], (0, 0, j, 0)) for j in range(self.conv_k))
            return self.c_proj(y)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        H, D = self.n_kv_head, self.head_dim
        q_gain, key_smear, query_smear = scalars.split([self.n_head * D, H * D, self.n_head * D])
        gq, gk = (2 * torch.sigmoid(self.smear_gate(x[..., :self.ve_gate_channels]))).view(B, T, 2, -1, 1).unbind(2)
        k = k + gk * key_smear.to(k.dtype).view(H, D) * F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))
        q = q + gq * query_smear.to(q.dtype).view(self.n_head, D) * F.pad(q[:, :-1], (0, 0, 0, 0, 1, 0))   # query smear
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # value residual (ResFormer): mix in the value embedding with an input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * (1 + q_gain).to(q.dtype).view(1, 1, self.n_head, D)   # per-head, per-channel query gain
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if window is None:          # full causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                       # sliding window: keys t-window..t (their FA3 window_size=(window, 0))
            y = flex_attention(q, k, v, block_mask=window)
        g = 2 * torch.sigmoid(F.linear(x[..., :self.ve_gate_channels], out_gate))      # [head gates | channel gates]
        y = y.transpose(1, 2) * g[..., :self.n_head].unsqueeze(-1)
        y = y.contiguous().view(B, T, -1) * g[..., self.n_head:]
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # grouped low-rank gate on the hidden units: one gate per group of 16 units, read from 32 channels
        self.group = 16
        self.hidden_gate = nn.Linear(32, 4 * config.n_embd // self.group, bias=False)

    def forward(self, x):
        h = F.relu(self.c_fc(x)).square()
        g = 2 * torch.sigmoid(self.hidden_gate(x[..., :32]))                          # (B, T, 128)
        h = (h.view(*h.shape[:-1], -1, self.group) * g.unsqueeze(-1)).view(h.shape)
        return self.c_proj(h)


class ParamHolder(nn.Module):
    def __init__(self, **tensors):
        super().__init__()
        for name, t in tensors.items():
            setattr(self, name, nn.Parameter(t))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window, out_gate, scalars):
        x = x + self.attn(norm(x), ve, cos_sin, window, out_gate, scalars)
        h = norm(x)
        x = x + self.mlp(h)
        return x, h


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
        # low-rank n-gram logit priors (folded into the head matmul):
        #   logits += U[x_t] V^T (rank 112)  +  U2[x_(t-1)] V2^T (rank 48, skip-bigram with its own factors)
        self.bigram_U = nn.Parameter(torch.zeros(config.vocab_size, 112))
        self.bigram_V = nn.Parameter(torch.zeros(config.vocab_size, 112))
        self.skip_U = nn.Parameter(torch.zeros(config.vocab_size, 48))
        self.skip_V = nn.Parameter(torch.zeros(config.vocab_size, 48))
        # token metadata embedding: (byte-length bucket, leading whitespace, first-character class) per token
        self.n_meta = 16 * 2 * 6
        self.register_buffer("token_meta", torch.zeros(config.vocab_size, dtype=torch.long), persistent=False)
        self.meta_embed = nn.Parameter(torch.zeros(self.n_meta, config.n_embd))
        self.skip_lambdas = nn.Parameter(torch.zeros(config.n_layer // 2))   # U-net skips: layer i -> layer L-1-i
        # causal per-head conv over each value embedding (contextual value embeddings), init identity
        self.ve_conv_k = 2
        self.n_ve = sum(has_ve(i, config.n_layer) for i in range(config.n_layer))
        self.ve_conv = nn.Parameter(torch.zeros(self.n_ve * self.ve_conv_k * config.n_kv_head))
        # selective lags: per token, head and lag gates (from x0[:32]) on the contextual value embedding
        self.ve_lag_gate = nn.Parameter(torch.zeros(self.n_ve, 32, self.ve_conv_k * config.n_kv_head))
        self.x0_conv_k = 4      # causal per-channel conv over the token embeddings (init: identity)
        self.x0_conv = nn.Parameter(torch.zeros(self.x0_conv_k * config.n_embd))
        # per-layer, per-token gate on the re-injected embedding, read from the residual stream
        self.x0_gate_channels = 32
        self.x0_gates = nn.Parameter(torch.zeros(config.n_layer, self.x0_gate_channels, 2))   # [x0 gate, residual gate]
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # packed per-layer extras (one tensor each, so each costs one gradient all-reduce, not n_layer):
        # gate_bank.attn.out_gate is an attention matrix stack (Muon per slice, as separate 2-D gates would be)
        self.gate_bank = nn.ModuleDict({"attn": ParamHolder(out_gate=torch.zeros(config.n_layer, config.n_head + config.n_embd, 32))})
        self.n_scalars = config.n_embd + kv_dim + 1 * config.n_embd
        self.attn_scalars = nn.Parameter(torch.zeros(config.n_layer * self.n_scalars))   # 1-D: AdamW, no decay
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
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            if block.attn.conv is not None:     # start as a uniform average of the last conv_k positions
                block.attn.conv.fill_(1.0 / block.attn.conv_k)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.hidden_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.ve_conv.zero_()
        self.ve_conv.view(self.n_ve, self.ve_conv_k, -1)[:, 0] = 1.0
        self.ve_lag_gate.zero_()
        self.x0_conv.zero_()
        self.x0_conv[:self.config.n_embd] = 1.0
        self.x0_gates.zero_()
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        torch.nn.init.normal_(self.bigram_U, std=1.0)
        self.bigram_V.zero_()
        torch.nn.init.normal_(self.skip_U, std=1.0)
        self.skip_V.zero_()
        self.gate_bank.attn.out_gate.zero_()
        self.attn_scalars.zero_()
        for block in self.transformer.h:
            if block.attn.smear_gate is not None:
                torch.nn.init.zeros_(block.attn.smear_gate.weight)
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
        for i, w in LOCAL_WINDOWS.items():
            windows[i] = w
        return [None if w is not None and w >= long_window else w for w in windows]

    def _mask(self, window, T, device):
        """flex_attention BlockMask of the sliding window (built once, outside torch.compile)"""
        key = (window, T, str(device))
        if key not in self._masks:
            def sliding(b, h, q_idx, kv_idx):     # multi-resolution heads: local heads + long-window heads
                w = torch.where(h < LOCAL_HEADS, min(LOCAL_HEAD_WINDOW, window), window)
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= w)
            self._masks[key] = create_block_mask(sliding, B=None, H=self.config.n_head, Q_LEN=T, KV_LEN=T, device=device)
        return self._masks[key]

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        e = (self.transformer.wte(idx) + F.embedding(self.token_meta[idx], self.meta_embed)).bfloat16()
        w = self.x0_conv.view(self.x0_conv_k, -1).bfloat16()
        x = norm(sum(w[j] * F.pad(e[:, :T - j], (0, 0, j, 0)) for j in range(self.x0_conv_k))).bfloat16()
        x0 = x      # bf16 residual stream (like every other activation)
        ve_conv = self.ve_conv.view(self.n_ve, self.ve_conv_k, -1).repeat_interleave(
            self.config.n_embd // self.config.n_kv_head, dim=-1).bfloat16()     # per head -> per channel
        scalars = self.attn_scalars.view(self.config.n_layer, self.n_scalars)
        skips = []
        # all VE tables' selective lag gates in one matmul: (B, T, n_ve, k, H)
        lag_gates = (2 * torch.sigmoid(x0[..., :32] @ self.ve_lag_gate.permute(1, 0, 2).reshape(32, -1).to(x0.dtype))
                     ).view(B, T, self.n_ve, self.ve_conv_k, -1)
        h = x0      # gate read: the most recent normalized state (layer i-1's MLP input; x0 for layer 0)
        for i, block in enumerate(self.transformer.h):
            g = 2 * torch.sigmoid(h[..., :self.x0_gate_channels] @ self.x0_gates[i])
            x = self.resid_lambdas[i] * g[..., 1:] * x + self.x0_lambdas[i] * g[..., :1] * x0
            if i >= self.config.n_layer // 2:      # second half: add the mirrored first-half output
                x = x + self.skip_lambdas[self.config.n_layer - 1 - i].to(x.dtype) * skips[self.config.n_layer - 1 - i]
            ve = None
            if str(i) in self.value_embeds:     # contextual value embedding, computed in bf16
                e = self.value_embeds[str(i)](idx).bfloat16()
                w = ve_conv[list(self.value_embeds.keys()).index(str(i))]
                vi = list(self.value_embeds.keys()).index(str(i))
                lg = lag_gates[:, :, vi].repeat_interleave(self.config.n_embd // self.config.n_kv_head, dim=-1)   # (B, T, k, C)
                ve = sum(lg[:, :, j] * w[j] * F.pad(e[:, :T - j], (0, 0, j, 0)) for j in range(self.ve_conv_k))
            w = self.windows[i]
            x, h = block(x, ve, cos_sin, None if w is None else self._mask(w, T, idx.device),
                         self.gate_bank.attn.out_gate[i], scalars[i])
            if i < self.config.n_layer // 2:
                skips.append(x)
        x = norm(x)
        softcap = 15
        # bigram prior folded into the head matmul: [x, U[x_t]] @ [W_out, V]^T (one matmul, no extra logits pass)
        prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        feats = torch.cat([x, F.embedding(idx, self.bigram_U).to(x.dtype), F.embedding(prev, self.skip_U).to(x.dtype)], dim=-1)
        logits = F.linear(feats, torch.cat([self.lm_head.weight, self.bigram_V, self.skip_V], dim=1)).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
        return logits

    def param_components(self):
        # component of every parameter: for the per-component size caps, and (with its shape) for the
        # harness's choice of optimizer
        components = {}
        for name, p in self.named_parameters():
            if name.startswith(("transformer.wte.", "value_embeds.", "meta_embed", "bigram_U", "skip_U")):
                components[name] = "embedding"
            elif name.startswith(("lm_head.", "bigram_V", "skip_V")):
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
def _token_meta(i):
    """bucketed byte length (0-15), leading whitespace flag, first non-space character class (6)"""
    try:
        b = tokenizer.enc.decode_single_token_bytes(i)
    except Exception:
        return 0
    ws = int(b[:1] in (b" ", b"\n", b"\t", b"\r"))
    core = b.lstrip(b" \n\t\r")
    c = core[:1]
    cls = 0 if not c else 1 if c.isupper() else 2 if c.islower() else 3 if c.isdigit() else 4 if c < b"\x80" else 5
    return (min(len(b), 15) * 2 + ws) * 6 + cls
model.token_meta.copy_(torch.tensor([_token_meta(i) for i in range(vocab_size)], device=device))
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
