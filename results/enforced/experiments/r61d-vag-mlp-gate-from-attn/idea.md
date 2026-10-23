# MLP write gate from the block's attention output
Parent: r58b-va-gates-adam (0.94618)
Type: explore

The MLP write gate from the MLP's own normalized input helped (r60b, -0.0016). Variant: the gate comes from what
attention just retrieved (normalized): m <- m * 2 sigmoid(U V norm(a)), rank 32, AdamW gate. (A per-token scalar
version of this, r14d, was very harmful.) Compare with r60b.
