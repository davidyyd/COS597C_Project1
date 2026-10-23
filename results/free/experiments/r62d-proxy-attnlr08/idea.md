# proxy (bf16 resid): Muon LR x0.8 for attention matrices only

Parent: r59a-proxy-bf16resid (0.76505). Muon LR was tuned jointly for attention and MLP matrices; split it: attention (q,k,v,proj,gates) x0.8, MLP unchanged.
