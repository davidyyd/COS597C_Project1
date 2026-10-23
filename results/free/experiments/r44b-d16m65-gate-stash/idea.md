# r43d + grad stash (bf16 flat buffer between micro-steps; memory)

Parent: r43d-d16m65-gate (best, 0.76693, 77.5GB). Memory diag (r43b): micro-step 1 runs with micro-step 0's fp32 grads (6.2GB) resident. Stash them into the bf16 all_reduce buffer and free them (they are reduced in bf16 anyway): ~-3GB peak. Same math up to one bf16 rounding.
