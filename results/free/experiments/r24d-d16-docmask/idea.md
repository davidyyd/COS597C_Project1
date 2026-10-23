# d16 (bigram262k, emb WD0) + document masking

Parent: r21d-d16-bigram262k-embwd0 + doc masking (r12d: -0.0011 at small scale, +12% time there). Measures the real cost/benefit at d16: per-batch BlockMasks (eager create_block_mask), all layers via flex_attention with same-document constraint.
