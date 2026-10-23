# Per-channel MLP write gate from the MLP's own (normalized) input
Parent: r58a-best-gates-adam
Type: explore

Per-token scalar multipliers on MLP writes were harmful (r03d, r14d), but those gates came from unbounded outputs
or from attention. Here the gate is per channel and computed from the MLP's bounded input through a rank-32 map
(labelled as a gate: AdamW): m <- m * 2 sigmoid(U V u), U init 0. A GLU on the output side of the MLP.
Confirm: < r58a by > 0.0005. Refute: worse (write-side gating keeps hurting).
