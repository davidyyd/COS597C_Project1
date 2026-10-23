# Grouped RMSNorm for block inputs
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: normalization of values (and separately of value embeddings) helped a lot: balancing the magnitudes
of the parts that get mixed matters. In the residual stream, a few loud directions (e.g. the re-injected
embedding or massive activations) can dominate norm(x) and drown the rest. Normalize block inputs per group of
128 channels (4 groups, each unit RMS) instead of over all 512, so every channel group contributes.
Parameter-free. Confirm: < r01b (0.95653) by > 0.001.
Closest work: GroupNorm (vision), RMSNorm. Differs: grouped RMSNorm as the pre-norm of a transformer block.
