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
torch._dynamo.config.cache_size_limit = 64
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
    n_layer: int = 16
    n_head: int = 14
    n_kv_head: int = 14
    n_embd: int = 1792
    window_pattern: str = "SL"


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


def _f(t):
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


class XSAGate(torch.autograd.Function):
    """z = (y - a * (<y,v>/<v,v>) v) * g   (learnable XSA followed by a head-wise output gate)
    y, v: [B,H,T,D]; a: [1,H,1,1]; g: [B,H,T,1]. Saves only y, v, a, g (y and v are saved by attention anyway)."""
    @staticmethod
    def forward(ctx, y, v, a, g):
        n = (_f(v) * _f(v)).sum(-1, keepdim=True) + 1e-6
        c = (_f(y) * _f(v)).sum(-1, keepdim=True) / n
        z = ((_f(y) - _f(a) * c * _f(v)) * _f(g)).to(y.dtype)
        ctx.save_for_backward(y, v, a, g)
        return z

    @staticmethod
    def backward(ctx, dz):
        y, v, a, g = ctx.saved_tensors
        yf, vf, af, gf, dzf = _f(y), _f(v), _f(a), _f(g), _f(dz)
        n = (vf * vf).sum(-1, keepdim=True) + 1e-6
        c = (yf * vf).sum(-1, keepdim=True) / n
        u = yf - af * c * vf
        dg = (dzf * u).sum(-1, keepdim=True)
        du = dzf * gf
        duv = (du * vf).sum(-1, keepdim=True)
        dy = du - af * (duv / n) * vf
        dv = -af * (c * du + duv * (yf / n - 2 * c * vf / n))
        da = -(c * duv).sum(dim=(0, 2, 3), keepdim=True)
        return dy.to(y.dtype), dv.to(v.dtype), da.to(a.dtype), dg.to(g.dtype)


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
        self.xsa_alpha = nn.Parameter(torch.zeros(self.n_head))
        self.out_gate = nn.Linear(32, self.n_head, bias=False)
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
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if window is None:          # full causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.12)
        else:                       # sliding window: keys t-window..t (their FA3 window_size=(window, 0))
            y = flex_attention(q, k, v, block_mask=window, scale=0.12)
        a = torch.tanh(self.xsa_alpha).view(1, -1, 1, 1)
        g = (2 * torch.sigmoid(self.out_gate(x[..., :32]))).transpose(1, 2).unsqueeze(-1).to(y.dtype)
        if torch.is_grad_enabled():     # fused XSA + head gate: saves only y, v (kept by attention anyway), a, g
            y = XSAGate.apply(y, v, a, g)
        else:
            a = a.to(y.dtype)
            coef = (y * v).sum(-1, keepdim=True) / ((v * v).sum(-1, keepdim=True) + 1e-6)
            y = (y - (a * coef) * v) * g
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class ReLU2Proj(torch.autograd.Function):
    """relu(h)^2 @ w.T that saves only h (not relu(h)^2) and recomputes the activation in backward"""
    @staticmethod
    def forward(ctx, h, w):
        ctx.save_for_backward(h, w)
        return F.relu(h).square() @ w.to(h.dtype).t()

    @staticmethod
    def backward(ctx, g):
        h, w = ctx.saved_tensors
        r = F.relu(h)
        g = g.to(h.dtype)
        dw = (g.flatten(0, -2).t() @ r.square().flatten(0, -2)).to(w.dtype)
        dh = (g @ w.to(h.dtype)) * (2 * r)
        return dh, dw


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 12800, bias=False)
        self.c_proj = nn.Linear(12800, config.n_embd, bias=False)

    def forward(self, x):
        h = self.c_fc(x)
        if torch.is_grad_enabled():
            return ReLU2Proj.apply(h.to(torch.bfloat16), self.c_proj.weight)
        return self.c_proj(F.relu(h).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.backin_mix = nn.Parameter(torch.zeros(1))
        self.canon_a = nn.Parameter(torch.zeros(config.n_embd, 2))
        self.canon_m = nn.Parameter(torch.zeros(config.n_embd, 2))

    def canon(self, h, w):
        # residual causal depthwise conv over time: h_t + sum_j w[:, j] * h_{t-j}, j = 0..3
        h, w = h.to(torch.bfloat16), w.to(torch.bfloat16)
        out = h + h * w[:, 0]
        for j in range(1, w.size(1)):
            out = out + F.pad(h[:, :-j], (0, 0, j, 0)) * w[:, j]
        return out

    def forward(self, x, ve, cos_sin, window, attn_in=None):
        if attn_in is None:
            h = x + 0 * self.backin_mix.to(x.dtype)
        else:      # last layers: attention reads (mostly) the cached 2/3-depth activation
            h = attn_in + self.backin_mix.to(x.dtype) * (x - attn_in)
        x = x + self.attn(self.canon(norm(h), self.canon_a), ve, cos_sin, window)
        x = x + self.mlp(self.canon(norm(x), self.canon_m))
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
        self.bigram_size = 262139
        self.bigram = nn.Embedding(self.bigram_size, config.n_embd)    # hashed (previous, current) token pairs
        self.register_buffer('sh31', torch.arange(31), persistent=False)
        self.register_buffer('sh29', torch.arange(29), persistent=False)
        self.register_buffer('ci31', torch.arange(config.n_embd) % 31, persistent=False)
        self.register_buffer('ci29', torch.arange(config.n_embd) % 29, persistent=False)
        self.bigram_out_size = 65521
        self.bigram_out = nn.Embedding(self.bigram_out_size, config.n_embd)    # output-side bigram table
        self.trigram_out = nn.Embedding(65521, config.n_embd)
        self.fourgram_out = nn.Embedding(65521, config.n_embd)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.backout_lambda = nn.Parameter(torch.zeros(1))
        self.skip_weights = nn.Parameter(torch.zeros(config.n_layer // 2))
        self.skip_gates = nn.ModuleList([nn.Linear(32, 1, bias=False) for _ in range(config.n_layer // 2)])
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
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.zeros_(self.bigram.weight)
        torch.nn.init.zeros_(self.bigram_out.weight)
        torch.nn.init.zeros_(self.trigram_out.weight)
        torch.nn.init.zeros_(self.fourgram_out.weight)
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
            block.attn.xsa_alpha.zero_()
            torch.nn.init.zeros_(block.attn.out_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.backout_lambda.fill_(0.0)
        self.skip_weights.fill_(0.5)
        for gl in self.skip_gates:
            torch.nn.init.zeros_(gl.weight)
        for block in self.transformer.h:
            block.canon_a.zero_()
            block.canon_m.zero_()
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
        inv_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4, dtype=torch.float32, device=device)
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(head_dim // 4)])
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()      # fp32 tables (precision is the harness's; they used bf16)
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        """per layer: None for full attention (L), else the sliding window (S: half the context); last layer full"""
        long_window, short_window = config.sequence_len, 256
        windows = [short_window if config.window_pattern.upper()[i % len(config.window_pattern)] == "S" else None
                   for i in range(config.n_layer)]
        windows[-1] = None
        return [None if w is not None and w >= long_window else w for w in windows]

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
        prev = F.pad(idx[:, :-1], (1, 0))
        pair = prev * 8192 + idx
        ha = (pair * 2654435761) % 2147483647
        hb = (pair * 40503 + 12345) % 2147483629
        bits = ((ha.unsqueeze(-1) >> self.sh31) & 1)[..., self.ci31] ^ ((hb.unsqueeze(-1) >> self.sh29) & 1)[..., self.ci29]
        sign = (1 - 2 * bits).to(torch.bfloat16)      # bigram sign trick (fixed +-1 pattern per bigram)
        x = norm(self.transformer.wte(idx) + self.bigram(pair % self.bigram_size) * sign)
        x = x.to(torch.bfloat16)      # bf16 residual stream (halves elementwise memory traffic)
        x0 = x
        n_half = self.config.n_layer // 2
        skips = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.config.n_layer - n_half:
                j = self.config.n_layer - 1 - i
                g = 2 * torch.sigmoid(self.skip_gates[j](x[..., :32]))
                x = x + (self.skip_weights[j] * g).to(x.dtype) * skips.pop()
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            w = self.windows[i]
            x = block(x, ve, cos_sin,
                      None if w is None else self._mask(w, T, idx.device),
                      x_backout if i >= self.config.n_layer - 5 else None)
            if i < n_half:
                skips.append(x)
            if i == (2 * self.config.n_layer) // 3 - 1:
                x_backout = x
        tri = (F.pad(idx[:, :-2], (2, 0)) * 8192 + prev) * 8192 + idx
        quad = ((F.pad(idx[:, :-3], (3, 0)) * 8192 + F.pad(idx[:, :-2], (2, 0))) * 8192 + prev) * 8192 + idx
        qa = (quad % 2147483647 * 2654435761) % 2147483647
        qb = (quad % 2147483629 * 40503 + 12345) % 2147483629
        qsign = ((1 - 2 * ((qa.unsqueeze(-1) >> self.sh31) & 1)).to(torch.bfloat16)[..., self.ci31]
                 * (1 - 2 * ((qb.unsqueeze(-1) >> self.sh29) & 1)).to(torch.bfloat16)[..., self.ci29])
        ta = (tri * 2654435761) % 2147483647
        tb = (tri * 40503 + 12345) % 2147483629
        tbits = ((ta.unsqueeze(-1) >> self.sh31) & 1)[..., self.ci31] ^ ((tb.unsqueeze(-1) >> self.sh29) & 1)[..., self.ci29]
        tsign = (1 - 2 * tbits).to(torch.bfloat16)
        x = norm(x - self.backout_lambda.to(x.dtype) * x_backout + self.bigram_out(pair % self.bigram_out_size) * sign
                 + self.trigram_out(tri % 65521) * tsign + self.fourgram_out(quad % 65521) * qsign)
        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
        return logits

    def param_components(self):
        # component of every parameter: for the per-component size caps, and (with its shape) for the
        # harness's choice of optimizer
        components = {}
        for name, p in self.named_parameters():
            if name.startswith(("transformer.wte.", "value_embeds.", "bigram.", "bigram_out.", "trigram_out.", "fourgram_out.")):
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
MUON_MU = 0.9
EMB_LR_MULT = 7.0
HEAD_LR_MULT = 0.4
WARMUP_FRAC = 0.02                    # linear warmup over the first 5% of steps, then linear decay to 0

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    # Polar Express coefficient schedule (Amsel et al. 2025)
    ns_consts = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]
    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile(dynamic=False)
def muon_update(grad, momentum, mu=0.95):
    grad = grad.to(momentum.dtype)
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp(momentum, mu)
    update = zeropower_via_newtonschulz5(update)
    # Moonshot/Kimi RMS-matching rule: match the per-parameter RMS of an AdamW update
    update *= 0.17 * max(grad.size(-2), grad.size(-1))**0.5
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
        # each rank orthogonalizes its share of the matrices (round-robin) into a zeroed flat bf16 buffer of
        # all updates; one all_reduce (sum) then gives every rank every update, and all ranks apply them
        # identically (an all_reduce runs ~5x faster here than the per-chunk all_gathers did)
        for group in self.param_groups:
            params = group["params"]
            if "offsets" not in group:
                group["offsets"] = [0]
                for q in params:
                    group["offsets"].append(group["offsets"][-1] + q.numel())
            offsets = group["offsets"]
            flat = torch.zeros(offsets[-1], dtype=torch.bfloat16, device=params[0].device)
            for i in range(rank, len(params), world_size):
                p = params[i]
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)
                if p.ndim == 2:
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                else:
                    stack = (-1, p.size(-2), p.size(-1))
                    update = muon_update(p.grad.reshape(stack), state["momentum"].view(stack),
                                         mu=group["mu"]).view_as(p)
                flat[offsets[i]:offsets[i + 1]].copy_(update.view(-1))
            dist.all_reduce(flat)
            for i, p in enumerate(params):
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(flat[offsets[i]:offsets[i + 1]].view_as(p), alpha=-group["lr"])

class AdamWBF16(torch.optim.Optimizer):
    """AdamW with exp_avg / exp_avg_sq stored in bf16 (math in fp32 per tensor); fp32 params and grads"""
    def __init__(self, params, lr, betas, eps, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            group["n"] = group.get("n", 0) + 1
            dev = group["params"][0].device
            # step-dependent scalars as 0-d tensors so the compiled update is not recompiled every step
            hp = torch.tensor([group["lr"], 1 - b1 ** group["n"], 1 - b2 ** group["n"]], device=dev)
            for p in group["params"]:
                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["v"] = torch.zeros_like(p, dtype=torch.bfloat16)
                adam_bf16_update(p, p.grad, st["m"], st["v"], hp, b1, b2, group["eps"], group["weight_decay"])


@torch.compile(dynamic=True)
def adam_bf16_update(p, g, m_bf, v_bf, hp, b1, b2, eps, wd):
    lr, bc1, bc2 = hp[0], hp[1], hp[2]
    m = m_bf.float() * b1 + g * (1 - b1)
    v = v_bf.float() * b2 + g * g * (1 - b2)
    m_bf.copy_(m)
    v_bf.copy_(v)
    p.copy_(p * (1 - lr * wd) - lr * (m / bc1) / ((v / bc2).sqrt() + eps))


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
    comp = model.param_components()
    name_of = {p: n for n, p in model.named_parameters()}
    emb = [p for p in adamw_params if comp[name_of[p]] == "embedding"]
    head = [p for p in adamw_params if comp[name_of[p]] == "head"]
    rest2d = [p for p in adamw_params if p.ndim >= 2 and comp[name_of[p]] not in ("embedding", "head")]
    adamw_groups = [dict(params=emb, weight_decay=0.0, lr=base_lr * EMB_LR_MULT),
                    dict(params=head, weight_decay=WEIGHT_DECAY, lr=base_lr * HEAD_LR_MULT),
                    dict(params=rest2d, weight_decay=WEIGHT_DECAY),
                    dict(params=[p for p in adamw_params if p.ndim < 2], weight_decay=0.0)]
    optimizers = [
        AdamWBF16([g for g in adamw_groups if g["params"]], lr=base_lr, betas=ADAM_BETAS, eps=1e-10),
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

LR = 1.2e-2                 # one learning rate for every parameter group
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens per optimizer step
DEVICE_BATCH_SIZE = 16    # sequences per GPU per forward/backward (8 GPUs x 32 x 2048 = 2**19)
SEED = 88

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
phase_times = {}
for step in range(NUM_STEPS):
    # no per-step host syncs: the CPU queues the next step while the GPU finishes this one
    if step == 11:
        torch.cuda.synchronize()
        t_train0 = time.time()
    if step % 100 == 0:
        torch.cuda.synchronize()
        t0 = time.time()
    timing = 50 <= step < 60
    def tick(name, _t=[0.0]):
        if timing:
            torch.cuda.synchronize()
            now = time.time()
            if name is not None:
                phase_times[name] = phase_times.get(name, 0.0) + now - _t[0]
            _t[0] = now
    tick(None)
    train_loss = torch.zeros((), device=device)
    stash = stash_views = None
    for micro_step in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        with autocast_ctx:
            loss = model(x, y, reduction="sum")      # summed token cross-entropy
        tick("forward")
        train_loss += loss.detach()
        loss.backward()
        tick("backward")
        if micro_step < grad_accum_steps - 1:
            # stash this micro-step's grads in the flat bf16 all_reduce buffer and free the fp32 grads, so they
            # do not sit in memory during the next micro-step's forward/backward
            plist = list(model.parameters())
            if stash is None:
                stash = torch.zeros(sum(p.numel() for p in plist), dtype=torch.bfloat16, device=device)
                stash_views = [t.view_as(p) for t, p in zip(stash.split([p.numel() for p in plist]), plist)]
            torch._foreach_add_(stash_views, [p.grad.to(torch.bfloat16) for p in plist])
            for p in plist:
                p.grad = None
    dist.all_reduce(train_loss)
    # sum gradients across GPUs: one all_reduce of a flat bf16 copy (half the bytes, no per-tensor latency)
    if step == 0:
        torch.cuda.empty_cache()    # release cached blocks so NCCL can allocate its buffers for the big all_reduce
    grads = [p.grad for p in model.parameters()]
    torch._foreach_add_(stash_views, [g.to(torch.bfloat16) for g in grads])   # + last micro-step (bf16 buffer)
    dist.all_reduce(stash)
    torch._foreach_copy_(grads, stash_views)
    del stash, stash_views, grads
    tick("grad_allreduce")
    set_hparams(optimizers, step)
    for opt in optimizers:
        opt.step()
        tick(type(opt).__name__)
    model.zero_grad(set_to_none=True)
    if step == 60:
        print("phase ms/step: " + ", ".join(f"{k}={v * 100:.1f}" for k, v in phase_times.items()), flush=True)

    if step % 100 == 0:
        train_loss_f = train_loss.item() / TOTAL_BATCH_SIZE
        # Fast fail: abort if loss is exploding or NaN (checked every 100 steps)
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            exit(1)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"step {step:05d} ({100 * step / NUM_STEPS:.1f}%) | loss: {train_loss_f:.6f} | lrm: {lr_multiplier(step):.2f} | dt: {dt*1000:.0f}ms", flush=True)

torch.cuda.synchronize()
total_training_time = time.time() - t_train0

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
