# Per-block torch.compile (compile time / overhead test) on r22a

Parent: r22a-d21-bigram-embwd0-bf16ar (0.77694, 81s compile at step 0, 3334s total). Efficiency: compiling the whole 21-layer model unrolls every layer (81s compile + another compile for eval). Compile each Block (identical code -> graphs reused across layers) and the head (norm+lm_head+softcap+CE) separately; the cheap glue (embeddings, x0/skip mixing) runs eager. Goal: ~60s less overhead to make room for the 262k table. Same math.
