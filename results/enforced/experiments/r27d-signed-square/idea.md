# Signed-square MLP activation h|h|
Parent: baseline
Type: explore

Hypothesis: ReLU^2 discards the negative half of each pre-activation. The odd function h|h| keeps the
quadratic growth of ReLU^2 but uses both signs, at the same cost (no extra parameters; the two-sided ReLU^2
with a per-unit weight, r11d, failed the step-time cap).
Confirm: < baseline by > 0.001. Refute: worse (the sparsity of ReLU^2 matters).
Closest work: ReLU^2 (Primer), polynomial / squared activations. Differs: signed square in a ReLU^2 model.
