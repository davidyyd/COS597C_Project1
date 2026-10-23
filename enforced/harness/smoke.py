"""
smoke.py -- FROZEN. The model check, step 1 of ar-check (check.py): checks the model file's source
(static_check.py: no file, network, process or gradient access), builds the model, runs a small
forward/backward (also under torch.compile), checks causality, and reports per-component params and
FLOPs/token against the caps in harness/caps.json, the precision rule (precision_check.py: fp32
parameters, no lower precision of the model's own), and the optimizer of every parameter. The step
time is checked by check.py's speed benchmark.

It always runs as a GPU task of the local queue (common/localgpu.py): started anywhere else, it submits
itself with a snapshot of the model file, waits, and prints the result.

usage: python harness/smoke.py [path/to/model.py] [--json out.json]
"""

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
SNAPSHOTS = paths.SNAPSHOTS

if not os.environ.get("GQ_TASK_ID"):
    # not a GPU task: run this same check as one, on a snapshot of the model file
    from pool_run import run_on_pool
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = Path(args[i + 1]).resolve()
        args = args[:i] + args[i + 2:]
    src = Path(args[0]).resolve() if args else Path(__file__).resolve().parent.parent / "model.py"
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    snap = SNAPSHOTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{src.parent.name}-{src.name}"
    shutil.copy(src, snap)
    cmd = f"python {Path(__file__).resolve()} {snap}" + (f" --json {out}" if out else "")
    print(f"checking {src.name} on one GPU", flush=True)
    sys.exit(run_on_pool(cmd, gpus=1, time_limit="20m", workdir=str(src.parent), name="smoke", model_file=snap))

import torch
from prepare import ROOT, SEQ_LEN, VOCAB_SIZE, build, component_params, count_params, flops_per_token, split_params

argv = sys.argv[1:]
json_out = None
if "--json" in argv:
    i = argv.index("--json")
    json_out = Path(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
model_file = Path(argv[0]) if argv else ROOT / "model.py"

def report(**fields):
    if json_out is not None:
        json_out.write_text(json.dumps(fields, indent=2) + "\n")
caps = json.loads((Path(__file__).resolve().parent / "caps.json").read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# before the model file is imported (importing runs its code): no file, network or process access
from static_check import violations as static_violations  # noqa: E402
static = [f"model file not allowed: {v}" for v in static_violations(model_file.read_text())]
if static:
    report(status="fail", violations=static)
    for v in static:
        print(f"VIOLATION: {v}")
    print("SMOKE: FAIL (model file)")
    sys.exit(1)

import precision_check  # noqa: E402
settings_before = precision_check.numeric_settings()   # before the model file is imported

try:
    torch.manual_seed(1337)
    model, wrapper = build(model_file, device)
    model.init_weights()
    params = count_params(model)
    components = component_params(model)
    mapping_c = model.param_components()
    muon, adamw, plan = split_params(model)
    from flex_dense import dense_flex   # flex_attention -> its dense equivalent for the two uncompiled checks
    with dense_flex(model):
        flops = flops_per_token(wrapper, device, batch=1)
    for name, p in model.named_parameters():
        assert p.grad is None, f"grad not cleared for {name}"
    with dense_flex(model):
        precision = precision_check.violations(model, settings_before, device, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    # every parameter must receive a gradient: the trainer requires it (unused parameters crash training)
    x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device, dtype=torch.int32)
    y = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device, dtype=torch.int64)
    wrapper(x, y).backward()
    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    model.zero_grad(set_to_none=True)
    # causality: the logits at position t may depend only on tokens 0..t of the same sequence.
    # Change the tokens after t (and, separately, the other sequences of the batch) and require the
    # logits up to t to stay the same. Tested on a copy with random weights (zero-initialized
    # projections would hide a leak) and both as written and compiled (training uses torch.compile;
    # attention kernels differ between them and between GPU types, so this runs on the training GPUs).
    import copy
    probe = copy.deepcopy(model)
    with torch.no_grad():
        for p in probe.parameters():
            p.normal_(0, 0.02) if p.ndim > 1 else p.normal_(1.0, 0.1)
    leaks = []
    gen = torch.Generator(device=device).manual_seed(0)
    rand = lambda *shape: torch.randint(0, VOCAB_SIZE, shape, device=device, dtype=torch.int32, generator=gen)
    for mode in ("eager", "compiled"):
        compiled = probe if mode == "eager" else torch.compile(probe, dynamic=False)
        def fwd(x, f=compiled):          # same precision as training (bf16 autocast)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                return f(x)
        try:
            with torch.no_grad():
                x = rand(8, SEQ_LEN)
                base = fwd(x).float()
                tol = 1e-2 * max(1.0, base.abs().max().item())
                for t in (0, 1, 2, 3, 7, 255, 511, 1023, 1535, SEQ_LEN - 2):
                    y = x.clone()
                    y[:, t + 1:] = rand(8, SEQ_LEN - 1 - t)
                    d = (fwd(y).float()[:, :t + 1] - base[:, :t + 1]).abs().max().item()
                    if d > tol:
                        leaks.append(f"[{mode}] logits at positions 0..{t} change by {d:.3g} when only later tokens change")
                y = x.clone()
                y[1:] = rand(7, SEQ_LEN)
                d = (fwd(y).float()[0] - base[0]).abs().max().item()
                if d > tol:
                    leaks.append(f"[{mode}] logits of one sequence change by {d:.3g} when other sequences in the batch change")
        except Exception as e:
            if mode == "eager":
                raise
            # a compile failure is reported by the compile check below
    del probe
    # training runs the model through torch.compile; some code runs fine as written but crashes the
    # compiler, so compile it here too and run one small forward/backward (takes ~1-2 minutes)
    compile_error = None
    if device.type == "cuda":
        try:
            wrapper.compile(dynamic=False)
            x = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device, dtype=torch.int32)
            y = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device, dtype=torch.int64)
            wrapper(x, y).backward()
            torch.cuda.synchronize()
        except Exception as e:
            compile_error = (type(e).__name__ + ": " + str(e)).strip().splitlines()
            compile_error = " | ".join(compile_error[-3:])[-600:]
        model.zero_grad(set_to_none=True)
except Exception:
    traceback.print_exc()
    report(status="crash", error=traceback.format_exc()[-4000:])
    print("SMOKE: FAIL (crash)")
    sys.exit(1)

violations = list(precision)
if leaks:
    violations.append("not causal (future tokens leak into the predictions): " + "; ".join(leaks))
if compile_error:
    violations.append(f"torch.compile (used by training) fails on this model: {compile_error}")
if no_grad:
    violations.append(f"{len(no_grad)} parameter(s) receive no gradient (unused parameters crash training; "
                      f"remove them or make them used): {no_grad[:10]}")
if flops > caps["flops_ratio_cap"] * caps["flops_per_token"]:
    violations.append(f"FLOPs per token {flops / caps['flops_per_token']:.3f}x baseline > cap {caps['flops_ratio_cap']}x")
for comp, n in components.items():
    if n > caps["component_caps"][comp]:
        violations.append(f"{comp} params {n:,} > cap {caps['component_caps'][comp]:,} "
                          f"(baseline {caps['baseline_components'][comp]:,})")

print(f"device: {device}")
print(f"params: total={params['total']:,}")
for comp, n in components.items():
    print(f"  {comp:<9} {n:>12,}  cap {caps['component_caps'][comp]:>12,}  "
          f"({n / max(caps['baseline_components'][comp], 1):.2f}x baseline)")
print(f"flops/token (fwd+bwd): {flops:.4e} ({flops / caps['flops_per_token']:.4f}x baseline; cap {caps['flops_ratio_cap']}x)")
print(f"optimizer (decided by the harness from component and shape): "
      f"muon={sum(p.numel() for p in muon):,} params in {len(muon)} tensors, "
      f"adamw={sum(p.numel() for p in adamw):,} params in {len(adamw)} tensors")
import re  # noqa: E402
groups = {}                                # parameters that differ only in layer index are listed once
for n, p in model.named_parameters():
    key = (re.sub(r"\.\d+\.", ".*.", n), tuple(p.shape), mapping_c[n], plan[n])
    groups[key] = groups.get(key, 0) + 1
for (n, shape, comp, opt), count in groups.items():
    slices = f" ({'x'.join(map(str, shape[:-2]))} matrices of {shape[-2]}x{shape[-1]})" if opt == "muon per slice" else ""
    wd = "" if len(shape) >= 2 else ", no weight decay"
    print(f"  {n:<32} {count:>3} x {str(list(shape)):<18} {comp:<9} -> {opt}{slices}{wd}")
print(f"torch.compile check: {'FAILED' if compile_error else ('ok' if device.type == 'cuda' else 'skipped (no GPU)')}")
print(f"step-time cap: {caps['step_ms_cap']:.1f} ms (checked next by ar-check's speed benchmark on 8 GPUs)")
print(f"causality check: {'FAILED' if leaks else 'ok'}")
print(f"precision check: {'FAILED' if precision else 'ok'}")
report(status="fail" if violations else "pass", violations=violations, no_grad_params=no_grad, causality_leaks=leaks,
       precision=precision,
       compile_error=compile_error,
       total_params=params["total"],
       components=components, flops_per_token=flops, flops_ratio=flops / caps["flops_per_token"],
       optimizer=plan)
if violations:
    for v in violations:
        print(f"VIOLATION: {v}")
    print("SMOKE: FAIL (caps)")
    sys.exit(1)
print("SMOKE: PASS")
