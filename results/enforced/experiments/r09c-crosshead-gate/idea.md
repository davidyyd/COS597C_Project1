# Cross-head low-rank self-gate
Parent: r08b-lowrank-selfgate
Type: exploit

Replace the per-head rank-32 gate by a rank-48 map across all heads' outputs: g = a.y + U(V y_flat),
V [48,512], U [512,48] (U init 0): a head can be switched off by what another head retrieved (e.g. a local
head defers to a retrieval head). Same param budget scale. Confirm: < r08b by > 0.0005.
