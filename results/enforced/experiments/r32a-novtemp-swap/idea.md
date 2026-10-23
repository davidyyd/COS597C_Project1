# Best model: swap the half-layer low-rank gate for a novelty temperature
Parent: r28a-best-outtemp
Type: exploit

Novelty temperature (nov = 1 - cos(x0_t, sum of earlier x0); r31c: -0.0022 vs r01b) next to the doc-position
temperature: q *= 1 + tanh(c_h f_doc + d_h nov). r28a has no time headroom (adding it measured 69.7 ms), so the
half-layer low-rank output gate (~2 ms, worth ~-0.0011) is removed. Prefix sums are computed as a matmul with a
strictly lower-triangular mask (cumsum version failed the compiled causality check numerically).
Confirm: < r28a (0.94335).
