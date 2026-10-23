"""
train.py -- FROZEN. Trains one model.py at one base LR and writes train.log + result.json.

usage: torchrun --standalone --nproc_per_node=1 harness/train.py \
           --model-file path/to/model.py --lr 3.16e-2 --out-dir path/to/dir [--seed 1337] [--mode train|speed]

--mode train (default): the full 2.0B-token training; the step time over steps 20-120 is measured and
  recorded (check_step_ms), not enforced.
--mode speed: the speed benchmark of ar-check: the same training steps, stopped after the
  measurement window (steps 20-120); result.json has status "speed" and check_step_ms. The step-time
  cap is enforced on this measurement, by ar-check.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.distributed as dist
from prepare import (BATCH_SIZE, MAX_MICROBATCH, SEQ_LEN, STEP_CHECK_WINDOW,
                     TRAIN_PATTERN, TRAIN_STEPS, VAL_PATTERN, VAL_TOKENS, build, build_optimizers,
                     count_params, distributed_data_generator, set_hparams, token_bytes)

MIN_MICROBATCH = 16

# testing only (never set by the runner): shrink the run to exercise the code path on small GPUs
MAX_MICROBATCH = int(os.environ.get("AR_TEST_MAX_MICROBATCH", MAX_MICROBATCH))
TRAIN_STEPS = int(os.environ.get("AR_TEST_TRAIN_STEPS", TRAIN_STEPS))

parser = argparse.ArgumentParser()
parser.add_argument("--model-file", required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--mode", choices=["train", "speed"], default="train")
args = parser.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
assert 8 % dist.get_world_size() == 0

logfile = out_dir / "train.log"
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

def write_result(**fields):
    if dist.get_rank() == 0:
        result = dict(model_file=str(args.model_file), lr=args.lr, seed=args.seed, **fields)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

print0(Path(args.model_file).read_text())
print0("=" * 100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0(f"model_file={args.model_file} base_lr={args.lr} seed={args.seed}", console=True)
print0("=" * 100)

local_sequences = BATCH_SIZE // dist.get_world_size() // SEQ_LEN
mbs = min(MAX_MICROBATCH, local_sequences)
val_inputs, val_targets = next(distributed_data_generator(VAL_PATTERN, VAL_TOKENS))
# bits per byte (their evaluate_bpb): nats over targets with >0 bytes / (ln 2 * bytes of all targets)
val_bytes_of = token_bytes()[val_targets]
val_weights = (val_bytes_of > 0).float()
val_bytes = val_bytes_of.sum().double()
dist.all_reduce(val_bytes, op=dist.ReduceOp.SUM)

torch.manual_seed(args.seed)
model, wrapper = build(args.model_file, device)
wrapper.compile(dynamic=False)
model.init_weights()
print0(f"params={json.dumps(count_params(model))}", console=True)

optimizers = build_optimizers(model, args.lr)
train_loader = distributed_data_generator(TRAIN_PATTERN, BATCH_SIZE)
for p in model.parameters():
    dist.broadcast(p.detach(), 0)

training_time = 0
last_val_step = 0
check_ms = None
val_loss = float("nan")
step_avg = float("nan")
dist.barrier()
t0 = time.perf_counter()
for step in range(TRAIN_STEPS + 1):

    # --------------- VALIDATION SECTION -----------------
    val_step_freq = 125 if step / TRAIN_STEPS < 0.9 else 25
    if step == TRAIN_STEPS or step % val_step_freq == 0:
        dist.barrier()
        time_since_last_val = time.perf_counter() - t0
        step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
        last_val_step = step
        training_time += time_since_last_val
        wrapper.eval()
        while True:
            val_loss = 0
            try:
                with torch.no_grad():
                    assert len(val_inputs) % mbs == 0
                    for i in range(len(val_inputs) // mbs):
                        val_loss += wrapper(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs],
                                            val_weights[i*mbs:(i+1)*mbs])
                break
            except torch.OutOfMemoryError:
                val_loss = None
                torch.cuda.empty_cache()
                if mbs <= MIN_MICROBATCH:
                    raise
                mbs //= 2
                print0(f"out of memory in validation: retrying with microbatch {mbs}", console=True)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = (val_loss.double() / (math.log(2) * val_bytes)).item()    # bits per byte
        print0(f"step:{step}/{TRAIN_STEPS} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        wrapper.train()
        if not math.isfinite(val_loss):
            write_result(status="diverged", step=step, val_loss=None, train_time=training_time)
            dist.destroy_process_group()
            sys.exit(0)
        dist.barrier()
        t0 = time.perf_counter()

    if step == TRAIN_STEPS:
        break

    # --------------- STEP TIME (enforced by ar-check on --mode speed) -----------------
    # steps 20..120 contain no validation (first one after 0 is at 125); compare against the baseline step time
    if step == STEP_CHECK_WINDOW[0]:
        t_check = time.perf_counter()
    elif step == STEP_CHECK_WINDOW[1]:
        check_ms = 1000 * (time.perf_counter() - t_check) / (STEP_CHECK_WINDOW[1] - STEP_CHECK_WINDOW[0])
        print0(f"step_time_check: {check_ms:.1f} ms/step over steps {STEP_CHECK_WINDOW[0]}-{STEP_CHECK_WINDOW[1]}",
               console=True)
        if args.mode == "speed":
            write_result(status="speed", step=step, val_loss=None, check_step_ms=check_ms, microbatch=mbs)
            dist.destroy_process_group()
            sys.exit(0)

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    while True:
        # If a microbatch does not fit in GPU memory, halve it and redo this step with more
        # accumulation. The summed loss and its gradient over the full batch are unchanged.
        assert len(inputs) % mbs == 0
        train_loss = torch.zeros((), device=device)
        try:
            for i in range(len(inputs) // mbs):
                microbatch_loss = wrapper(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
                train_loss += microbatch_loss.detach()
                microbatch_loss.backward()
            break
        except torch.OutOfMemoryError:
            microbatch_loss = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if mbs <= MIN_MICROBATCH:
                raise
            mbs //= 2
            print0(f"out of memory at step {step}: retrying with microbatch {mbs} "
                   f"({len(inputs) // mbs} accumulation steps)", console=True)
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    train_loss /= BATCH_SIZE
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    set_hparams(optimizers, step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{TRAIN_STEPS} train_loss:{train_loss:.5f}"
           + f" train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=(step % 100 == 0))
    if not torch.isfinite(train_loss):
        print0(f"non-finite train loss at step {step+1}", console=True)
        write_result(status="diverged", step=step + 1, val_loss=None, train_time=approx_training_time)
        dist.destroy_process_group()
        sys.exit(0)

write_result(status="ok", step=TRAIN_STEPS, val_loss=val_loss, train_time=training_time,
             step_avg_ms=1000 * training_time / TRAIN_STEPS, check_step_ms=check_ms, microbatch=mbs)
dist.destroy_process_group()
