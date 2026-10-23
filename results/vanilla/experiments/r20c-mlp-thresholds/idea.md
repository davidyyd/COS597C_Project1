# Learned per-unit thresholds in the ReLU^2 MLP
Parent: r18a-bf16-rope (the baseline with bf16 RoPE tables; faster and slightly better control). Type: explore.

Hypothesis: with no biases, every ReLU^2 unit fires exactly when its input projection is positive, so
about half the units are active for any input and sparsity cannot be tuned per unit. A learned threshold
per hidden unit (relu(W x - b)^2) lets units become selective (sparser, feature-detector-like) or broad.
Confirm: < 0.95739 (r18a). Refute: no gain (Muon-trained matrices already shift pre-activations).
Closest work: biases in MLPs (standard in GPT-2), ReLU^2 (Primer), learned activation thresholds (JumpReLU
in SAEs). Difference: thresholds in a bias-free ReLU^2 LM MLP.
