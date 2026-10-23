# Per-token MLP input temperature after the gainless pre-norm
Parent: baseline
Type: explore

Hypothesis: the pre-norm (no gain) gives every token a unit-RMS MLP input. ReLU^2 is 2-homogeneous, so the
input scale sets both which units are near their kink and the output magnitude (x s^2). A bounded per-token
input scale s = 1 + tanh(w . norm(x)) in (0, 2), computed from the (bounded) normalized input, restores the
magnitude channel the norm erased — the same principle as the query temperature that helped (r02c).
Unlike the failed MLP self-gate (r03d), the gate is driven by a unit-RMS input, not by the unbounded output.

Change: MLP(u) with u <- u * (1 + tanh(w . u)), w in R^512 per layer, init 0.
Confirm: < baseline by > 0.001.
Closest work: RMSNorm gains (static per channel), "scale" in nGPT; here per-token, input-dependent.
