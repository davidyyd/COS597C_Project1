# Head-gate biases from the current AND the previous token
Parent: r76c-token-headgate
Type: explore
Hypothesis: r76c's per-token head-gate bias helped; which heads are useful plausibly depends on the local bigram
context, not only the current token (e.g. after a preposition). An additive bias from the previous token's table
gives the gate a cheap bigram signal without a bigram table.
Change: g_h += B[token_t, l, h] + B'[token_{t-1}, l, h]; B' init 0 (8192 x 32).
Confirm: below r76c by > 0.0003.
Closest work: bigram hash embeddings (residual adds, modded-nanogpt); here factorized unigram biases on head gates.
