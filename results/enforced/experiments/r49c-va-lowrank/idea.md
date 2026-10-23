# Value-path baseline + scalar and low-rank head self-gate (re-measure)
Parent: r43a-base-valuepath
Type: exploit

y_h *= 2 sigmoid(a_h . y_h + U_h V_h y_h) (r10d: -0.0048 on the old baseline). Confirm: < r43a.
