# Per-channel gate on the token-identity re-injection
Parent: r58b-va-gates-adam
Type: explore

The x0 re-injection uses one scalar per layer (or one per token with the x0 gate). Let the current state choose
which channels of the token identity to re-inject: x <- r x + a (2 sigmoid(U V norm(x))) * x0 (rank 16 per layer,
AdamW gates, init 1). Confirm: < r58b by > 0.001.
