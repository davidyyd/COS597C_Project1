# proxy: learnable per-channel gain on the final norm

Parent: r45d. Explore: all RMSNorms are gain-free; a learnable gain on the final norm (init 1, AdamW no-WD) lets the model rescale channels before the head (the lm_head could do it, but Adam on a small vector adapts differently).
