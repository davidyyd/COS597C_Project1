# Per-head output normalization (x per-head scale) only in layer 3 (the middle global layer)
Parent: r94d-vag-ynorm-early
Type: explore (mechanism)
On the cheap base, output normalization in layers 0-3 gave -0.0018 but in 0-2 only ~0 (r95d): the gain sits in the
global layer 3 (late 4-7 incl. global layer 7: -0.0004). Normalize only layer 3 (the middle global layer).
Confirm: below r83a (0.94006) and r94d (0-3, 0.93831).
