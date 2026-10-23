# Replace layer-0 attention by a causal depthwise convolution over the values
Parent: baseline. Type: explore.

Hypothesis: the first layer sees raw (re-normalized) token embeddings; its attention can only match
token identities and mostly builds local n-gram features (previous-token mixing). A causal depthwise
convolution of width 4 over the value projection does local mixing directly and drops the q/k
projections and the attention of layer 0, freeing time for other changes.

Change: layer 0: y = c_proj(sum_{j<4} w_j ⊙ v_{t-j}), v = c_v(x); no q, k, attention (w init 1/4).
Confirm: ≈ or < baseline with less time. Refute: clearly worse (layer 0 needs long-range attention).
Closest work: Canon layers (Allen-Zhu 2025), H3 shift-SSM, hybrid conv-attention stacks. Difference:
replacing the first attention entirely with a value-convolution.
