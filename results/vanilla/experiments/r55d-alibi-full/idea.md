# Fixed ALiBi recency biases in the two full-attention layers (with RoPE)
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: the global layers attend over up to 2048 keys with only RoPE for position; a fixed geometric
set of per-head recency slopes (one sharply local head ... one almost flat head) gives them explicit
multi-scale distance priors (no learnable parameters in the score_mod, so it stays fast).
Confirm: < 0.95726 (r25a). Closest work: ALiBi (Press et al. 2022); RoPE+ALiBi hybrids.
