# FP8 matmuls for all block linears (speed test at d16w1024)

Parent: r02c-d16w1024-grouplr-lr7e3. Explore (efficiency). Wall clock is the binding constraint on model size; FP8 (per-tensor dynamic scaling, e4m3 forward, e5m2 grads, torch._scaled_mm) on attention + MLP linears could give ~1.3x speed -> bigger model in 60 min. lm_head stays bf16. Compare val_bpb to 0.84038 and train time to 1254s.
