# d20 canon-bf16 at device batch 16 via compile activation memory budget 0.8 (+ win256)

Parent: r12a-d20-canonbf16-b8-pe + win256. Explore (efficiency). Batch 16 is ~5% faster than 8 but d20 canon needs ~79GB at b16; torch._functorch.config.activation_memory_budget=0.8 lets the min-cut partitioner recompute cheap ops (norms, canon, relu^2) to cut saved activations. Risk: OOM or recompute overhead.
