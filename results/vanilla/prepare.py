"""
prepare.py -- read-only. Fixed constants, tokenizer, dataloader and evaluation for train.py.

Karpathy's autoresearch prepare.py, reduced to what train.py uses (no download / tokenizer-training code:
the data and tokenizer are already prepared). Two changes from his file:
  * the training budget is a fixed number of optimizer steps (NUM_STEPS) instead of 5 minutes of wall clock;
  * training runs on 8 GPUs (torchrun): each rank reads its own share of every batch, and evaluate_bpb
    evaluates its share of the validation rows and sums across ranks (the same metric as his).
The rows are his BOS-aligned best-fit packing of ClimbMix, precomputed once (identical to what his
make_dataloader produces, checked row by row) and held in memory.
"""

import math
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist

MAX_SEQ_LEN = 2048          # context length
NUM_STEPS = 3815            # training budget: optimizer steps of 2**19 tokens (~2.0B tokens)
EVAL_TOKENS = 40 * 524288   # number of tokens for val eval
VOCAB_SIZE = 8192
BOS_TOKEN = "<|reserved_0|>"

DATA_DIR = "/scratch/gpfs/DANQIC/yy8435/autoresearch/nanogpt-ar/data/climbmix"
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")


class Tokenizer:
    """Minimal tokenizer wrapper."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        else:
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    return torch.from_numpy(np.load(os.path.join(TOKENIZER_DIR, "token_bytes.npy"))).to(device)


def _rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rows(split):
    path = os.path.join(DATA_DIR, f"{split}_rows.bin")
    header = np.fromfile(path, dtype=np.int32, count=256)
    nrows, row = int(header[2]), int(header[3])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(nrows, row))


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """BOS-aligned rows (best-fit packed), B rows of T+1 tokens per call on this rank; yields
    (inputs, targets, epoch). Rank r of W takes rows r*B .. r*B+B-1 of every block of W*B rows, so the
    W ranks together read the same rows in the same order as one GPU reading W*B rows at a time."""
    assert split in ["train", "val"] and T == MAX_SEQ_LEN
    rank, world = _rank_world()
    rows = _rows(split)
    per = B * world
    n = len(rows) // per
    starts = np.arange(n) * per + rank * B
    mine = torch.from_numpy(np.ascontiguousarray(rows[(starts[:, None] + np.arange(B)[None, :]).reshape(-1)])
                            .astype(np.int64)).view(n, B, T + 1)
    epoch, i = 1, 0
    while True:
        if i == n:
            epoch, i = epoch + 1, 0
        buf = mine[i].pin_memory()
        i += 1
        yield (buf[:, :-1].to("cuda", non_blocking=True), buf[:, 1:].to("cuda", non_blocking=True), epoch)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums. Each rank evaluates its share of the EVAL_TOKENS; the sums are added
    across ranks.
    """
    rank, world = _rank_world()
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN * world)
    total_nats = torch.zeros((), dtype=torch.float64, device="cuda")
    total_bytes = torch.zeros((), dtype=torch.float64, device="cuda")
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().double()
        total_bytes += nbytes.sum().double()
    if world > 1:
        dist.all_reduce(total_nats)
        dist.all_reduce(total_bytes)
    return (total_nats / (math.log(2) * total_bytes)).item()
