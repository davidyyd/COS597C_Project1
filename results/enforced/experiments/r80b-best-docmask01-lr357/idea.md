# Best + document masking in local layers 0-1, low-rank head gate in 3, 5, 7
Parent: r77b-token-head-mlp
Type: exploit
Same hypothesis as r80a with a cheaper split: mask only layers 0-1, keep more of the low-rank head gate.
Confirm: below r77b by > 0.0005.
