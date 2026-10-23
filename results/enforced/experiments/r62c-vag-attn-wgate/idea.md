# Attention write gate from its own input (fast value-path base)
Parent: r58b-va-gates-adam (0.94618)
Type: explore

a <- a * 2 sigmoid(U V norm(x)) (rank 32, AdamW). Compare with the MLP write gate (r60b, -0.0016) and with gated
attention. Confirm: < r58b by > 0.001.
