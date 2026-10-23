# Per-dimension value-mixing gates
Parent: r58b-va-gates-adam (0.94618)
Type: explore

Value path with per-dimension gates through a shared rank-16 map from the input (AdamW gates):
v = s (G1 * norm(W_v u) + G2 * norm(VE)), so each value channel decides how much context vs token identity it
carries. (On the best model it measured 69.1 ms.) Confirm: < r58b.
