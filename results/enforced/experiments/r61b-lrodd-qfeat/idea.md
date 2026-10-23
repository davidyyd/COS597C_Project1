# Low-rank gate in odd layers + per-dimension query gate
Parent: r59a-lr-all-other (0.93278)
Type: exploit (time swap)

r59a has no step-time headroom. Restrict the low-rank self-gate to the odd layers (as in r56b/r58a) and spend the
time on a gate that helped on the fast value-path base (r60: MLP write gate -0.0016, query feature gate -0.0012).
Confirm: < r59a.
