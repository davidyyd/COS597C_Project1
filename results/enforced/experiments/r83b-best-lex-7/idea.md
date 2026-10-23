# Best + lexical-match channel in the last (global) layer
Parent: r81a-dm012-lr357
Type: exploit
Hypothesis: the lexical-match channel (q, k += per-head weight x normalized token-embedding slice) gave -0.0004 on
the fast base in all layers but costs ~0.7 ms per layer. The last, global layer is where copying from anywhere in
the row matters most; add it only there.
Confirm: below r81a (mean 0.93009) by > 0.0003.
