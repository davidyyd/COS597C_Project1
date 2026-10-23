# Per-head value normalization
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: value magnitudes vary per token (projection of the normalized input plus value embeddings), so a
few loud values dominate attention outputs regardless of attention weights. Normalize values per head
(v_h <- s_h norm(v_h), s_h learned, init 1): each attended token contributes a unit direction and the output
magnitude measures agreement among attended values, which the head self-gate can read.
Confirm: < r01b (0.95653) by > 0.001. (Per-token value scales hurt, r06a; this removes value magnitude.)
Closest work: QK-norm (normalizes q and k only), nGPT (normalizes everything). Differs: value-only norm.
