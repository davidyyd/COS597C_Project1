# Self-gated queries (per channel, after projection)
Parent: r58b-va-gates-adam (0.94618)
Type: explore

q <- q * 2 sigmoid(U V q) per channel (rank 32, AdamW gate) before RoPE/QK-norm: queries select their own features.
Compare with the q/k input gate (r67a, -0.0018) and the self-gated values (r68b, -0.0018).
