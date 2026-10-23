# Self-gated keys (per channel, after projection)
Parent: r58b-va-gates-adam
Type: explore

k <- k * 2 sigmoid(U V k) per channel before RoPE/QK-norm. (Per-token key scales hurt earlier; this is a per-channel
self-selection followed by QK-norm.) Compare with r69b.
