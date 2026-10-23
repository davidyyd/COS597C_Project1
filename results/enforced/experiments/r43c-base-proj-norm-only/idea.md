# Only normalize the values after mixing (ablation)
Parent: baseline
Type: explore (ablation)

The other half of the value path: v <- s_h norm(W_v u + gate * VE) (the r34d change, -0.0016 on r01b). On the
plain baseline. Compare with r43a/r43b.
