# Multiplicative bigram code in the contextual x0 (on the contextual-x0 model)
Parent: r11b-context-x0. Type: explore.

Hypothesis: additive lag mixing (contextual x0) cannot represent token pairs; e_t ⊙ e_{t-1} gives a
pair-specific code without hashing or extra tables. The hashed bigram table (r01a) failed, so this tests
whether bigram identity helps when it comes for free from the existing embedding.
Change: x0 = norm(sum_j w_j ⊙ e_{t-j} + u ⊙ e_t ⊙ e_{t-1}), u per channel, init 0.
Confirm: < 0.9533 (r11b). Refute: no gain.
Closest work: hashed n-gram embeddings (Engram, BLT); bilinear/multiplicative interactions (mLSTM,
multiplicative RNNs). Difference: product of the two existing token embeddings as the bigram feature.
