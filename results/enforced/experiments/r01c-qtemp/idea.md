# Per-token query temperature after QK-norm
Parent: baseline
Type: explore

Hypothesis: QK-norm (without gains) fixes every query to the same norm, so all tokens attend with the same
sharpness. Some tokens need sharp, retrieval-like attention, others broad averaging. Restore a per-token,
per-head temperature computed from the input.

Change: q_h <- norm(q_h) * exp(w_h . x), W [heads, 512] init 0.

Confirm: val_loss < baseline by > 0.002. Refute: no gain / instability.
Closest work: learned QK-norm gains (static), scalable softmax (log-n scaling, position-based),
selective attention temperature. Differs: input-dependent temperature per token and head after QK-norm.
