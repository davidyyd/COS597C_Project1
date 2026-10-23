"""
flex_dense.py -- FROZEN. flex_attention cannot run uncompiled under the FLOP counter or the precision
check's dispatch mode (a higher-order op without rules there). For those two checks only, smoke.py runs
the model with flex_attention replaced by this exact dense equivalent: the BlockMask's mask_mod evaluated
on the full [q, kv] grid (and the score_mod, if any) with plain matmul/softmax attention. Training, the
speed benchmark and the compiled causality check use the real flex_attention. Model files therefore call
flex_attention directly and never branch on compilation (torch.compiler is not allowed in them).
"""

import contextlib
import math
import sys

import torch
import torch.nn.functional as F
import torch.nn.attention.flex_attention as fa

REAL = fa.flex_attention


def dense_flex_attention(query, key, value, score_mod=None, block_mask=None, scale=None, enable_gqa=False,
                         return_lse=False, kernel_options=None, **kwargs):
    B, Hq, L, E = query.shape
    S = key.shape[-2]
    if enable_gqa and key.shape[1] != Hq:
        key = key.repeat_interleave(Hq // key.shape[1], dim=1)
        value = value.repeat_interleave(Hq // value.shape[1], dim=1)
    scale = 1.0 / math.sqrt(E) if scale is None else scale
    dev = query.device
    b = torch.arange(B, device=dev)[:, None, None, None]
    h = torch.arange(Hq, device=dev)[None, :, None, None]
    qi = torch.arange(L, device=dev)[None, None, :, None]
    ki = torch.arange(S, device=dev)[None, None, None, :]
    if score_mod is None and not return_lse:
        mask = None if block_mask is None else block_mask.mask_mod(b, h, qi, ki).expand(B, Hq, L, S)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)
    s = (query @ key.transpose(-2, -1)) * scale
    if score_mod is not None:
        s = score_mod(s, b, h, qi, ki)
    if block_mask is not None:
        s = s.masked_fill(~block_mask.mask_mod(b, h, qi, ki), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = torch.softmax(s, dim=-1) @ value
    return (out, lse) if return_lse else out


@contextlib.contextmanager
def dense_flex(model):
    """inside: flex_attention, as reached from the model's module (its globals or torch's module), is the
    dense equivalent"""
    mod = sys.modules.get(type(model).__module__)
    patched = [(fa, "flex_attention")]
    if mod is not None:
        patched += [(mod, k) for k, v in vars(mod).items() if v is REAL]
    for obj, name in patched:
        setattr(obj, name, dense_flex_attention)
    try:
        yield
    finally:
        for obj, name in patched:
            setattr(obj, name, REAL)
