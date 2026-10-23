#!/bin/bash
# FROZEN. One GPU task (common/localgpu.py) = one model.py at one base LR on all of the task's GPUs (8 by default, one
# node, torchrun DDP; same global batch, tokens and schedule as on 1 GPU). Output goes to OUT_DIR
# (train.log, result.json; the task log is OUT_DIR/task.log).
# usage (by the runner): localgpu.submit("bash harness/run_train.sh MODEL_FILE LR OUT_DIR [MODE] [SEED]", gpus=8, ...)
# MODE: train (default, the full training) or speed (ar-check's step-time benchmark, steps 0-120)
set -eo pipefail
MODEL_FILE="$1"; LR="$2"; OUT_DIR="$3"; MODE="${4:-train}"; SEED="${5:-1337}"   # SEED: initialization only
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HARNESS/env.sh"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoid OOM from allocator fragmentation
export OMP_NUM_THREADS=8
export TMPDIR=/tmp/$USER-$GQ_TASK_ID; mkdir -p "$TMPDIR"   # local temp files
trap 'rm -rf "$TMPDIR"' EXIT
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "task=$GQ_TASK_ID host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES lr=$LR mode=$MODE seed=$SEED start=$(date --iso-8601=seconds)"
nvidia-smi -L
torchrun --standalone --nproc_per_node="$NGPU" "$HARNESS/train.py" \
    --model-file "$MODEL_FILE" --lr "$LR" --out-dir "$OUT_DIR" --mode "$MODE" --seed "$SEED"
echo "end=$(date --iso-8601=seconds)"
