# Sourced by every command of this repository (the launchers, the harness and each GPU task).
# Edit the first block for your machine; the rest has defaults that work from a fresh clone.

# 1. The Python environment with PyTorch (see README: "Setup"). Activate it here, e.g.
#      source "$HOME/venvs/autoresearch/bin/activate"
#      conda activate autoresearch
if [ -n "${AR_ACTIVATE:-}" ]; then eval "$AR_ACTIVATE"; fi

# 2. Locations (defaults: inside the clone).
export AR_REPO="${AR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export AR_DATA="${AR_DATA:-$AR_REPO/data/climbmix}"      # tokenizer/, train_rows.bin, val_rows.bin
export AR_WORK="${AR_WORK:-$AR_REPO/work}"               # run folders, traces, caches, task logs
# export AR_GPUS=8                                        # GPUs to use (default: all that nvidia-smi lists)

# 3. Caches and threads.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER-triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/$USER-cuda-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$AR_WORK" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
