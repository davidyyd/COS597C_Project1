# Deep token embedding (token-wise MLP before re-injection)
Parent: baseline
Type: explore

Hypothesis: the embedding x0 re-injected before every block (x0 lambdas) is a lookup table, i.e. a linear
function of the token. A small token-wise ReLU^2 MLP applied once, x0 = norm(e + MLP(e)) (hidden 512, c_proj
init 0 so it starts as the identity), gives every layer non-linear token features at ~1% extra FLOPs.
Confirm: < baseline by > 0.001. Refute: no gain (layer 0 already computes token features).
Closest work: factorized/deep embeddings (ALBERT factorization, "embedding MLPs"). Differs: the refined
embedding is the one re-injected at every layer.
