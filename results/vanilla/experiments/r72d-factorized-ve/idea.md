# Factorised value embeddings (rank-256 tables + up-projection)
Parent: r25a-bf16-resid. Type: explore.
Value-embedding tables are the largest parameter block (4 x 4.2M). If a rank-256 factorisation keeps
their benefit, half of their parameters can be reallocated (e.g. to the n-gram logit priors, whose
returns per parameter are steep). Confirm: ≈ 0.95726 with 8.4M fewer parameters (42.5M).
