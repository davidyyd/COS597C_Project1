"""
precision_check.py -- FROZEN. Run-time half of the precision rule (static_check.py is the source half):
precision is the harness's. Parameters and buffers are fp32 (or integer/bool), and the model's own
arithmetic is fp32: the harness's bf16 autocast alone decides which ops run in lower precision.

Checked by smoke.py (ar-check's model check):
  * every floating parameter and buffer is fp32;
  * a forward and backward with autocast OFF produce no lower-precision or quantized tensor and call no
    low-precision kernel: without autocast a dtype-agnostic model computes only in fp32, so any bf16 /
    fp16 / fp8 / int8 tensor there was made by the model itself;
  * the global numeric settings (TF32, reduced-precision reductions, fp32 matmul precision, default
    dtype) are the same after the model is built and run as before it was imported.
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

LOW = {torch.bfloat16, torch.float16, torch.int8, torch.uint8, torch.complex32}
LOW |= {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
                                    "float8_e8m0fnu", "float4_e2m1fn_x2", "qint8", "quint8", "qint32", "quint4x2")
        if hasattr(torch, n)}
LOW_KERNELS = ("_scaled_mm", "_int_mm", "int8pack", "int4pack", "quantize", "fbgemm")


def numeric_settings():
    """the global settings that change numerics without a cast"""
    return dict(
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        default_dtype=str(torch.get_default_dtype()),
    )


def _tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


class _Recorder(TorchDispatchMode):
    """records every op that takes or makes a lower-precision tensor, or is a low-precision kernel"""
    def __init__(self):
        super().__init__()
        self.found = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func.overloadpacket).replace("aten.", "")
        low = sorted({str(t.dtype).replace("torch.", "") for t in _tensors((args, kwargs, out)) if t.dtype in LOW})
        if low or any(k in name for k in LOW_KERNELS):
            key = f"{name} ({', '.join(low) or 'low-precision kernel'})"
            self.found[key] = self.found.get(key, 0) + 1
        return out


def violations(model, settings_before, device, vocab_size=50257, seq_len=1024):
    """list of precision-rule violations of a built, initialized model (empty if none)"""
    out = []
    bad = [f"{n} ({str(p.dtype).replace('torch.', '')})" for n, p in model.named_parameters()
           if p.dtype != torch.float32]
    bad += [f"buffer {n} ({str(b.dtype).replace('torch.', '')})" for n, b in model.named_buffers()
            if b.is_floating_point() and b.dtype != torch.float32 or b.dtype in LOW]
    if bad:
        out.append(f"parameters and buffers must be fp32 (or integer/bool): {', '.join(bad[:8])}"
                   + (f" and {len(bad) - 8} more" if len(bad) > 8 else ""))
    x = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.int32)
    y = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.int64)
    rec = _Recorder()
    error = None
    try:
        with torch.autocast(device.type, enabled=False), rec:
            logits = model(x)
            torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).backward()
    except Exception as e:      # runs under autocast but not without it: it depends on lower precision
        error = (type(e).__name__ + ": " + str(e)).strip().splitlines()[0][:300]
    model.zero_grad(set_to_none=True)
    if rec.found:
        ops = "; ".join(f"{k} x{v}" for k, v in list(rec.found.items())[:6])
        out.append("with autocast off the model still computes in lower precision (it casts or quantizes "
                   f"by itself; precision is the harness's): {ops}")
    if error:
        out.append(f"the model fails with autocast off ({error}): it mixes dtypes by itself; a model must "
                   "run in fp32 when the harness's autocast is off")
    after = numeric_settings()
    changed = {k: (settings_before[k], after[k]) for k in after if after[k] != settings_before[k]}
    if changed:
        out.append("the model changes global numeric settings: "
                   + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items()))
    return out
