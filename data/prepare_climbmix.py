#!/usr/bin/env python3
"""
prepare_climbmix.py -- the data of Karpathy's autoresearch (github.com/karpathy/autoresearch, prepare.py at
228791f), written once for the harness:

  1. tokenizer: their BPE (rustbpe, vocab 8192 incl. 4 special tokens, GPT-4-style split pattern), trained
     exactly as theirs on the first 1e9 characters of the training shards (documents capped at 10k chars);
     token_bytes (UTF-8 length of each token, 0 for special tokens) for bits-per-byte;
  2. rows: their BOS-aligned best-fit packing (every row of T+1 = 2049 tokens starts with BOS; documents
     are packed largest-that-fits first, and when none fits the shortest one is cropped; 128-document
     tokenizer batches; a 1000-document buffer). Same algorithm and document order as their
     make_dataloader, computed once instead of on the fly, so every run reads the same rows:
       train_rows.bin  TRAIN_ROWS rows from the training shards (sorted, without the pinned val shard)
       val_rows.bin    VAL_ROWS rows from the pinned validation shard 06542 (their EVAL_TOKENS)

  bash data/download.sh && python data/prepare_climbmix.py [--workers 32]      # writes into $AR_DATA
Shards are downloaded by data/download.sh into $AR_DATA/parquet/.
"""

import argparse
import bisect
import collections
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("AR_DATA", Path(__file__).resolve().parent / "climbmix"))
PARQUET = DATA / "parquet"
TOK_DIR = DATA / "tokenizer"
CACHE = DATA / "tokens"

# their constants (prepare.py)
VOCAB_SIZE = 8192
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"
VAL_FILENAME = "shard_06542.parquet"
MAX_SEQ_LEN = 2048
EVAL_TOKENS = 40 * 524288
TOKENIZER_BATCH = 128
BUFFER_SIZE = 1000

# the harness's training length (prepare.TRAIN_STEPS x BATCH_SIZE tokens)
TRAIN_ROWS = 3815 * (524288 // MAX_SEQ_LEN)
VAL_ROWS = EVAL_TOKENS // MAX_SEQ_LEN
ROW = MAX_SEQ_LEN + 1
MAGIC, VERSION = 20261009, 2


def train_shards():
    return sorted(p for p in PARQUET.glob("shard_*.parquet") if p.name != VAL_FILENAME)


def train_tokenizer():
    """exactly their train_tokenizer()"""
    import pyarrow.parquet as pq
    import rustbpe
    import tiktoken
    pkl, tb = TOK_DIR / "tokenizer.pkl", TOK_DIR / "token_bytes.npy"
    if pkl.exists() and tb.exists():
        print("tokenizer: already trained", flush=True)
        return
    TOK_DIR.mkdir(parents=True, exist_ok=True)

    def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
        nchars = 0
        for path in train_shards():
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                for text in pf.read_row_group(rg).column("text").to_pylist():
                    doc = text[:doc_cap] if len(text) > doc_cap else text
                    nchars += len(doc)
                    yield doc
                    if nchars >= max_chars:
                        return

    t0 = time.time()
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(text_iterator(), VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    specials = {name: len(ranks) + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="rustbpe", pat_str=tok.get_pattern(), mergeable_ranks=ranks, special_tokens=specials)
    pkl.write_bytes(pickle.dumps(enc))
    nbytes = [0 if enc.decode([i]) in SPECIAL_TOKENS else len(enc.decode([i]).encode("utf-8")) for i in range(enc.n_vocab)]
    np.save(tb, np.array(nbytes, dtype=np.int32))
    test = "Hello world! Numbers: 123. Unicode: 你好"
    assert enc.decode(enc.encode_ordinary(test)) == test
    print(f"tokenizer: trained in {time.time() - t0:.0f}s, vocab {enc.n_vocab}", flush=True)


def tokenize_shard(path):
    """one shard -> its tokenizer batches (row group by row group, 128 documents each), BOS first in every document"""
    import pyarrow.parquet as pq
    out = CACHE / (Path(path).stem + ".npz")
    if out.exists():
        return str(out)
    enc = pickle.loads((TOK_DIR / "tokenizer.pkl").read_bytes())
    bos = enc.encode_single_token(BOS_TOKEN)
    pf = pq.ParquetFile(path)
    toks, lens, batch_ends = [], [], []
    for rg in range(pf.num_row_groups):
        docs = pf.read_row_group(rg).column("text").to_pylist()
        for i in range(0, len(docs), TOKENIZER_BATCH):
            for ids in enc.encode_ordinary_batch(docs[i:i + TOKENIZER_BATCH], num_threads=4):
                toks.append(np.array([bos] + ids, dtype=np.uint16))
                lens.append(len(ids) + 1)
            batch_ends.append(len(lens))
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, tokens=np.concatenate(toks), lens=np.array(lens, dtype=np.int64),
             batch_ends=np.array(batch_ends, dtype=np.int64))
    os.replace(tmp, out)
    return str(out)


class Docs:
    """their _document_batches: tokenizer batches in shard order (one pass; no epoch wrap is needed)"""
    def __init__(self, shards):
        self.shards, self.i, self.batches = list(shards), 0, collections.deque()

    def next_batch(self):
        while not self.batches:
            if self.i == len(self.shards):
                sys.exit("ran out of documents: download more shards")
            with np.load(CACHE / (self.shards[self.i].stem + ".npz")) as z:     # read each array once
                tokens, lens, batch_ends = z["tokens"], z["lens"], z["batch_ends"]
            self.i += 1
            offs = np.concatenate([[0], np.cumsum(lens)])
            start = 0
            for end in batch_ends:
                self.batches.append([tokens[offs[j]:offs[j + 1]] for j in range(start, end)])
                start = end
        return self.batches.popleft()


def pack(shards, nrows, path):
    """their make_dataloader's best-fit packing, row by row. The buffer is kept as {length: documents in
    insertion order} with a sorted list of lengths: 'the largest document that fits' (first such in buffer
    order on ties) and 'the shortest document' (first on ties) are then the same choices as their linear scans."""
    docs = Docs(shards)
    by_len, lens, size, seq = collections.defaultdict(collections.deque), [], 0, 0

    def add(doc):
        nonlocal size, seq
        n = len(doc)
        if not by_len[n]:
            bisect.insort(lens, n)
        by_len[n].append((seq, doc))
        seq += 1
        size += 1

    def take(n):
        nonlocal size
        _, doc = by_len[n].popleft()
        if not by_len[n]:
            lens.pop(bisect.bisect_left(lens, n))
        size -= 1
        return doc

    out = np.empty((nrows, ROW), dtype=np.uint16)
    t0 = time.time()
    for r in range(nrows):
        pos = 0
        while pos < ROW:
            while size < BUFFER_SIZE:
                for d in docs.next_batch():
                    add(d)
            remaining = ROW - pos
            k = bisect.bisect_right(lens, remaining) - 1
            if k >= 0:
                doc = take(lens[k])
                out[r, pos:pos + len(doc)] = doc
                pos += len(doc)
            else:
                doc = take(lens[0])
                out[r, pos:pos + remaining] = doc[:remaining]
                pos += remaining
        if r % 100_000 == 0:
            print(f"  {path.name}: row {r:,}/{nrows:,} ({time.time() - t0:.0f}s)", flush=True)
    header = np.zeros(256, dtype=np.int32)
    header[:4] = (MAGIC, VERSION, nrows, ROW)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(out.tobytes())
    os.replace(tmp, path)
    print(f"{path.name}: {nrows:,} rows of {ROW} tokens, {docs.i} shard(s) read", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    train_tokenizer()
    CACHE.mkdir(parents=True, exist_ok=True)
    shards = train_shards()
    print(f"tokenizing {len(shards)} training shards and the val shard", flush=True)
    with Pool(a.workers) as pool:
        for p in pool.imap(tokenize_shard, [str(s) for s in shards] + [str(PARQUET / VAL_FILENAME)]):
            print(f"  tokenized {Path(p).name}", flush=True)
    pack([PARQUET / VAL_FILENAME], VAL_ROWS, DATA / "val_rows.bin")
    pack(shards, TRAIN_ROWS, DATA / "train_rows.bin")


if __name__ == "__main__":
    main()
