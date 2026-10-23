# Best model: deep embedding swapped for bigram-hashed token MLP biases (layers 0-3)
Parent: r38a-gated-proj
Type: exploit (swap at equal step time)

Both give the MLPs non-linear token features; the bigram-hashed neuron bias (r45d: -0.0021 on r43a) also
carries the previous token. Confirm: < r38a (0.93717).
