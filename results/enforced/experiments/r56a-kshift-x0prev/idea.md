# Keys also read the previous token's deep embedding
Parent: r54a-gate37-k-shift-only (0.93555)
Type: exploit

Prev-token keys were an early win (induction shortcut) but marginal later. In the key-only-shift model the keys
have their own input; add the previous token's (deep, unit-RMS) embedding x0[t-1] with a scalar weight.
Confirm: < r54a.
