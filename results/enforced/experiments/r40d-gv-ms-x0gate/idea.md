# Gated-value simple model + multi-scale heads + x0 gate
Parent: r39d-base-gated-values
Type: exploit

Add two cheap components of the best model (multi-scale head windows in S layers; per-token x0 re-injection
gate) to r39d. Confirm: < r39d (0.94631) by > 0.001.
