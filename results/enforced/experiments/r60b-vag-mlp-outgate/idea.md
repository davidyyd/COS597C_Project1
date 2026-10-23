# Value-path baseline + per-channel MLP write gate from the MLP input (AdamW gate)
Parent: r58b-va-gates-adam
Type: explore

m <- m * 2 sigmoid(U V u) (rank 32; gave -0.0007 on r58a, too slow to stack there). Measure on the simple model.
Confirm: < r58b by > 0.001.
