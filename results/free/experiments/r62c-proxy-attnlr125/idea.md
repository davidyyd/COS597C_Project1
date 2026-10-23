# proxy (bf16 resid): Muon LR x1.25 for attention matrices only

Parent: r59a-proxy-bf16resid (0.76505). Muon LR was tuned jointly for attention and MLP matrices; split it: attention (q,k,v,proj,gates) x1.25, MLP unchanged.
