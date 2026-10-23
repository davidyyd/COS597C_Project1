# Small shared value-embedding table for the non-VE layers
Parent: r39d-base-gated-values
Type: explore

Pair-shared full VE tables hurt, but token identity in values helps. Use the spare embedding budget (1.05M) for
one 8192x128 table shared by the 4 non-VE layers, broadcast to all heads and mixed with the same gated
normalized path: v = s(g1 norm(W_v u) + g2 norm(small_ve[t])). Confirm: < r39d (0.94631).
