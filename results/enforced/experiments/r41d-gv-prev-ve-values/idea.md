# Previous token's value embedding as a third value source
Parent: r39d-base-gated-values
Type: explore

With gated normalized mixing, value sources can be combined safely. Add the previous token's value-embedding row
as a third source in VE layers: v = s (g1 norm(W_v u) + g2 norm(VE[t]) + g3 norm(VE[t-1])), so an attended
position also reports which token preceded it (bigram identity), useful for induction-like copying.
(Unnormalized, indexing VE by t-1 hurt: r13d.) Confirm: < r39d (0.94631) by > 0.001.
