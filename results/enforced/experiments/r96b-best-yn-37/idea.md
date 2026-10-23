# Per-head output normalization (x per-head scale) only in the global layers 3 and 7
Parent: r94a-best-blockdiag
Type: exploit
On the cheap base, output normalization in layers 0-3 gave -0.0018 but in 0-2 only ~0 (r95d): the gain sits in the
global layer 3 (late 4-7 incl. global layer 7: -0.0004). Normalize only the global layers 3 and 7.
Confirm: below r94a mean (0.92887).
