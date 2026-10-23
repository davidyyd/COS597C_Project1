# x0 gate + temperature + smeared keys
Parent: r02a-x0gate-temp (adds r02d-smeared-keys). Type: exploit.

Hypothesis: smeared keys (-0.0034 alone) help a different mechanism (induction heads) than the
x0 gate / temperature, so it should add.

Change: r02a + k_t <- k_t + a ⊙ k_{t-1}.
Confirm: < 0.9519. Closest work: smeared keys (Olsson et al. 2022).
