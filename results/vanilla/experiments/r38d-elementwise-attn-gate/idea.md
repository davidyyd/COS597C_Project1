# Elementwise (per-channel) attention-output gate from a 32-channel read
Parent: r25a-bf16-resid. Type: explore.

The per-head output gate was one of the largest single wins (r05c, -0.0054). A per-channel gate (low-rank:
32 channels -> 512 gates, init neutral) can modulate individual value features, not just whole heads.
Confirm: < 0.95726, and compare with the headwise gate's gain.
Closest work: gated attention (Qiu et al. 2025: elementwise SDPA-output gates). Difference: low-rank
32-channel read for cost.
