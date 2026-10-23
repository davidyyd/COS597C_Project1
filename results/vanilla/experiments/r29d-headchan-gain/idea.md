# Per-(head, channel) query gain after QK-norm (temperature and RoPE-band weighting in one)
Parent: r25a-bf16-resid. Type: explore.

Generalises r01d (per-head temperature, -0.0041) and r28a (per-channel gain shared by heads, -0.0027):
q <- norm(q) ⊙ G_h with G ∈ R^{4x128} per layer, init 1.
Confirm: < 0.954566 (r28a).
