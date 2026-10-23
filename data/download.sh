#!/bin/bash
# Download the ClimbMix shards used by Karpathy's autoresearch (huggingface.co/datasets/karpathy/climbmix-400b-shuffle):
# training shards 0..47 and the pinned validation shard 06542, about 4.2 GB, into $AR_DATA/parquet/.
#   source env.sh && bash data/download.sh && python data/prepare_climbmix.py
set -e
source "$(dirname "$0")/../env.sh"
mkdir -p "$AR_DATA/parquet" && cd "$AR_DATA/parquet"
{ seq 0 47; echo 6542; } | xargs -P 8 -I{} bash -c 'f=$(printf "shard_%05d.parquet" {}); [ -f $f ] && exit 0
  for a in 1 2 3 4 5; do curl -sSfL -o $f.tmp "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/$f" && mv $f.tmp $f && exit 0; sleep $((2**a)); done
  echo "FAILED $f"; exit 1'
echo "downloaded $(ls shard_*.parquet | wc -l) shards into $AR_DATA/parquet (expected 49)"
