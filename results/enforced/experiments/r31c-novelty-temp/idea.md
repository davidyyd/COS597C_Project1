# Attention temperature from token novelty
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: whether the current token is new in the context or a repeat should change how heads attend
(repeats -> sharp copying/induction; novel tokens -> broad context gathering). A parameter-free novelty
signal: nov_t = 1 - cos(x0_t, sum_{s<t} x0_s) (similarity to the running mean of earlier embeddings), used as a
per-head temperature q *= 1 + tanh(c_h nov_t), c init 0. Cheap ([B,T] features).
Confirm: < r01b (0.95653) by > 0.001.
Closest work: doc-position temperature (r19d), log-n scaling. Differs: content-novelty-driven temperature.
