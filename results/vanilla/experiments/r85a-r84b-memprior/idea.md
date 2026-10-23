# Memory read reused as a trigram-context logit prior
Parent: r84b-r83a-noctx-mlpgate2. Type: exploit.
The bigram and skip-bigram priors are additive in x_t and x_{t-1}, so they cannot express their
interaction. The hashed memory already gathers a 128-d vector keyed by (x_{t-1}, x_t) at layer 4; feed its
first 64 features also straight into the folded head (logits += m[:64] V_m^T): a hashed trigram prior at
no extra gather. Paid in parameters by cutting the skip-bigram rank 96 -> 64 (prior gates regrouped to 16x16).
Confirm: < 0.923372. Closest: hashed n-gram logit tables; Engram; ours shares one table between a
residual injection and an output prior.
