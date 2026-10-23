"""
model.py -- the ONLY file the autoresearch agent edits.
Baseline (branch karpathy-arch): the GPT of Karpathy's autoresearch (github.com/karpathy/autoresearch,
train.py @ 228791f, itself from nanochat) at the size its runs train (DEPTH = 8: 8 layers, width 512,
4 heads of 128):
  * token embedding, RMS-normalized (RMSNorm without gains everywhere), kept as x0; before every block
    x <- resid_lambda_i * x + x0_lambda_i * x0 (per-layer scalars, init 1 and 0.1)
  * pre-norm blocks; attention with separate q, k, v projections, RoPE (base 10000), then QK-norm
    (no gains), sliding windows in the pattern SSSL (S: half the context, 1024 tokens; L: full; the last
    layer always full)
  * value embeddings on alternating layers (1, 3, 5, 7): a per-token table added to v with a per-head
    gate 2 * sigmoid(W x[..., :32]) (W init 0)
  * ReLU^2 MLP with hidden width 4 * 512
  * untied LM head after a final RMSNorm; logits soft-capped at 15 (15 * tanh(logits / 15))
  * their initialization: embedding N(0, 1), LM head N(0, 0.001), q/k/v/fc uniform +-sqrt(3/d),
    output projections 0, value embeddings uniform +-sqrt(3/d), gates 0
Differences from their file, all from the harness: parameters stay fp32 under bf16 autocast (they cast the
embeddings and RoPE tables to bf16); attention runs on flex_attention / SDPA (they use FlashAttention 3;
same math); the
loss is computed by the harness from the returned logits.

Contract with the frozen harness (see CLAUDE.md):
  build_model(vocab_size)            -> nn.Module
  model(inputs[B, T] int32)          -> logits[B, T, vocab_size]
  model.init_weights()               -> initializes all parameters (called once, under the seed)
  model.param_components()           -> {param name: 'attention'|'mlp'|'embedding'|'head'|'other'}
The harness picks each parameter's optimizer from its component and shape (fixed rule):
  2-D attention/mlp                  -> Muon on the whole matrix
  3-D attention/mlp [n, rows, cols]  -> Muon on each [rows, cols] slice (e.g. one head)
  everything else                    -> AdamW
"""
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
        self.out_gate = nn.Parameter(torch.zeros(self.n_head, self.head_dim))
        self.q_temp = nn.Linear(self.n_embd, self.n_head, bias=False)
        self.k_temp = nn.Linear(self.n_embd, self.n_kv_head, bias=False)
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
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # QK-norm removes magnitude: restore a bounded per-token, per-head sharpness in (0, 2)
        q = q * (1 + torch.tanh(self.q_temp(x))).unsqueeze(-1)
        # and a per-key salience: how strongly each key's match/mismatch counts
        k = k * (1 + torch.tanh(self.k_temp(x))).unsqueeze(-1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if window is None:          # full causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                       # sliding window: keys t-window..t (their FA3 window_size=(window, 0))
            y = flex_attention(q, k, v, block_mask=window)
        y = y.transpose(1, 2)                              # B, T, H, D
        # self-gate: each head can suppress its own output depending on what it retrieved
        y = y * (2 * torch.sigmoid((y * self.out_gate).sum(-1, keepdim=True)))
        y = y.contiguous().view(B, T, -1)
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
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.q_temp.weight)
            torch.nn.init.zeros_(block.attn.k_temp.weight)
        for block in self.transformer.h:
            block.attn.out_gate.zero_()
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
        cos, sin = freqs.cos(), freqs.sin()      # fp32 tables (precision is the harness's; they used bf16)
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        """per layer: None for full attention (L), else the sliding window (S: half the context); last layer full"""
        long_window, short_window = config.sequence_len, config.sequence_len // 2
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

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
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
        return softcap * torch.tanh(logits / softcap)

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


def build_model(vocab_size: int) -> nn.Module:
    return GPT(GPTConfig(vocab_size=vocab_size))
