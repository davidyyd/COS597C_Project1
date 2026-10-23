# Self-gated values (per-channel, from the values themselves)
Parent: r58b-va-gates-adam (0.94618)
Type: explore

Self-gating from the gated tensor itself keeps working (inputs, head outputs); gates from other signals hurt. Test
values: v <- v * 2 sigmoid(U V v) per channel (rank 32, AdamW) after the value path. Compare with r66d (value-input
gate, -0.0006).
