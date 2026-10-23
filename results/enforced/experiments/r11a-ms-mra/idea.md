# Best-gate model + MLP reads the fresh attention output
Parent: r09a-gate-multiscale
Type: exploit

Add u = norm(x + a) + beta_l norm(a) (MLP reads the block's attention output undiluted; r10c: -0.0052 on
baseline) to r09a (0.95006). Confirm: < 0.9495.
