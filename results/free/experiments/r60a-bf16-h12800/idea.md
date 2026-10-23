# r59b (bf16 residual) with MLP hidden 12800 (100*128)

Parent: r59b-r50b-bf16resid (0.76419, 3392s total). bf16 residual saved ~100s; spend half on MLP width (12544 -> 12800, +2%) keeping margin for slow nodes (r59c timed out at ~+100s).
