# Best model + MLP reads the token identity
Parent: r09a-gate-multiscale
Type: exploit

Add the MLP-only x0 read (u = norm(x) + beta_l x0, beta init 0; r09d: -0.0022 on baseline) to the best
model. Confirm: < r09a (0.95006) by > 0.0005.
