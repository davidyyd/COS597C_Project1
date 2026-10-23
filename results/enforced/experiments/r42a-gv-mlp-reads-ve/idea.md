# MLP reads the normalized value embedding (value-path world)
Parent: r39d-base-gated-values
Type: explore (re-test in the fixed-VE model)

MLP reading the VE hurt earlier (r28d) when VE magnitudes were unnormalized. In VE layers let the MLP read the
token's normalized value embedding: u = norm(x) + beta norm(VE[t]) (beta init 0). Confirm: < r39d (0.94631).
