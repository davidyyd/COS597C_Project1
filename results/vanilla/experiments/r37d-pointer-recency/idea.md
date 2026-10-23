# Pointer with a learned log-distance recency bias
Parent: r36d-pointer-chunked. Type: explore.

Hypothesis: the most useful copy sources are usually recent (cache models: recency matters); with QK-norm
the pointer can only express recency through RoPE-free content scores. A learned scalar r with
S_ts -= r log(t - s) gives a power-law recency prior (r init 0).
Confirm: < 0.953060 (r36d). Closest work: neural cache models (Grave et al. 2017), ALiBi-style biases.
