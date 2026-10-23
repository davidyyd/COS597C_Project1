# proxy (bf16 resid): AdamW eps 1e-8

Parent: r59a-proxy-bf16resid (0.76505). eps 1e-10 with bf16 moments; grads are summed (large), so eps is negligible either way unless second moments underflow in bf16 for rare embedding rows; check 1e-8.
