# Pointer head with exact n-gram match priors (neural pointer + induction cue)
Parent: r36d-pointer-chunked (pointer of r34c, chunked). Type: explore.

Hypothesis: the pointer must learn to match contexts through its learned q/k projections; exact token
identity is the strongest induction cue and is free to compute. Adding learned bonuses to the pointer
score when the earlier token equals the current one (b1 [x_s = x_t]) and when the bigram also matches
(b2 [x_s = x_t and x_{s-1} = x_{t-1}]) turns the pointer into a neural, soft n-gram cache: it can copy
"what followed the last occurrence of my current bigram" from the start of training.
Change: S_ts += b1 m1_ts + b2 m2_ts (b init 0, learned). Strictly causal (only tokens <= t are used).
Confirm: < r36d. Closest work: n-gram caches / unbounded n-gram LMs (infini-gram), induction heads,
pointer sentinel models. Difference: exact n-gram match features as priors inside a learned pointer.
