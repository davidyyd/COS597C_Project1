# Best model + causal bag-of-tokens context channel
Parent: r05b-best-x0gate
Type: exploit

Add c_t = norm(sum_{s<=t} x0_s), re-injected before every block with a scalar b_i (init 0) — r06c gave
-0.0014 on baseline. Confirm: < r05b by > 0.0005.
