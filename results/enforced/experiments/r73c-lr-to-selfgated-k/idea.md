# Low-rank head gate -> self-gated keys (odd layers)
Parent: r64a-ingate-instead
Type: exploit

Hypothesis: self-gated keys (k *= 2σ(U V k), rank 32, AdamW) gave -0.0012 on the fast base (r69c). Keys select
their own channels, i.e. what aspects a token advertises. Swap it in for the low-rank output self-gate, same layers.
Confirm: below r64a (0.93169) and r73a.
Closest work: gated attention (Qiu et al. 2025: gates on the output); here a self-gate on the keys.
