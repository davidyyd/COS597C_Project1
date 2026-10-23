# Selective (token-gated) query/key smear
Parent: r31d-bigram-qk. Type: explore.

Hypothesis: whether a position's query/key should absorb its predecessor depends on the token (inside a
multi-piece word the predecessor is part of the same unit; after punctuation it is not). A per-token,
per-head gate on the smear coefficients (read from 32 channels, init neutral) makes the smear selective.
Change: q_t <- q_t + g^q_h(x_t) b ⊙ q_{t-1}, k_t <- k_t + g^k_h(x_t) a ⊙ k_{t-1}, g = 2 sigmoid(W x_t[:32]).
Confirm: < 0.953065 (r31d). Closest work: selective SSMs / data-dependent token shift (RWKV-6 "ddlerp").
