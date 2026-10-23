# Centered ReLU^2 (per-token adaptive threshold)
Parent: baseline
Type: explore

Hypothesis: with ReLU^2 the number of active hidden units per token is uncontrolled and drifts during
training (dead units, or dense activations). Subtracting each token's mean pre-activation before the ReLU
makes roughly half the units active around a per-token adaptive threshold, a parameter-free normalization
of sparsity.

Change: h = W1 u; h <- h - mean_j(h_j); out = W2 relu(h)^2.
Confirm: < baseline by > 0.001. Refute: worse (the mean carries useful signal / fixed threshold is better).
Closest work: LayerNorm before activation (center + scale), "CReLU", TopK / k-sparse activations. Differs:
only centering, parameter-free, inside the MLP.
