# Token-identity re-injection modulated by novelty
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: a novel token (unlike the context so far) carries information the context cannot supply, so its
identity should be re-injected more; a repeated/expected token less. x0 lambda_l *= 1 + tanh(c_l nov_t),
nov_t = 1 - cos(x0_t, mean of earlier x0) (parameter-free), c init 0. Confirm: < r01b (0.95653) by > 0.001.
Closest work: x0 lambdas (static), doc-modulated x0 (r21d), content x0 gate (r04d).
