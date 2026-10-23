# Value-path baseline + per-dimension query feature gate (AdamW gate)
Parent: r58b-va-gates-adam
Type: exploit (re-measure)

q *= 2 sigmoid(U_h V x) (rank 16), trained as a gate. Confirm: < r58b by > 0.0005.
