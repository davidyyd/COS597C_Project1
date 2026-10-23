#!/bin/bash
# FROZEN (karpathy-vanilla). One GPU task = one agent train.py on all of the task's GPUs (torchrun; 8 in our runs).
# usage (by ar-exp, through common/localgpu.py): bash harness/run_vanilla.sh TRAIN_PY OUT_DIR
# The run directory gets the agent's train.py and a fresh copy of the harness's prepare.py (read-only for
# the agent; its own copy is never used). stdout+stderr -> OUT_DIR/run.log; the summary lines train.py
# prints (val_bpb, training_seconds, ...) -> OUT_DIR/result.json.
set -eo pipefail
TRAIN_PY="$1"; OUT_DIR="$2"
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS/env.sh"
set -u
export OMP_NUM_THREADS=8   # more threads per rank oversubscribes the CPUs with 8 ranks (~4x slower steps)
export TMPDIR=/tmp/$USER-$GQ_TASK_ID-$GQ_ATTEMPT; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
WORK="$OUT_DIR/work"; mkdir -p "$WORK"
cp "$TRAIN_PY" "$WORK/train.py"; cp "$HARNESS/vanilla/prepare.py" "$WORK/prepare.py"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "task=$GQ_TASK_ID attempt=$GQ_ATTEMPT host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date --iso-8601=seconds)"
cd "$WORK"
rc=0
torchrun --standalone --nproc_per_node="$NGPU" train.py > "$OUT_DIR/run.log" 2>&1 || rc=$?
python - "$OUT_DIR" "$rc" <<'EOF'
import json, re, sys
out, rc = sys.argv[1], int(sys.argv[2])
log = open(f"{out}/run.log", errors="replace").read().replace("\r", "\n")
res = {"exit_code": rc}
for k in ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent", "total_tokens_M",
          "num_steps", "num_params_M", "depth"):
    m = re.findall(rf"^{k}:\s*(\S+)", log, re.M)
    if m:
        try:
            res[k] = float(m[-1])
        except ValueError:
            res[k] = m[-1]
json.dump(res, open(f"{out}/result.json", "w"), indent=2)
EOF
echo "end=$(date --iso-8601=seconds) rc=$rc"
