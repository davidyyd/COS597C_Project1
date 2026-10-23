# Value residual with normalized mixing
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: normalizing the value parts before a gated mix was a large gain for value embeddings (r36b/r37c).
Apply the same principle to a value residual: each layer's values = norm(W_v u) + tanh(W_g x) norm(v_{l-1}),
where v_{l-1} are the previous layer's projected values (signed per-head gate, 0 at init).
Confirm: < r01b (0.95653) by > 0.001.
Closest work: value residual learning (ResFormer: adds layer-1 values, fixed/learned scalar). Differs: previous
layer's values, normalized parts, token-dependent signed gate.
