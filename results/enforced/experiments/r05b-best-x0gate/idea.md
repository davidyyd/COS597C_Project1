# Best model + state-dependent x0 re-injection
Parent: r03b-qtanh-prevtok
Type: exploit

Add x <- r_i x + a_i (1 + tanh(u_i . norm(x))) x0 (r04d: -0.0026 on baseline) to the best model.
Confirm: < 0.9520.
