# Best with the low-rank head gate in block-diagonal matmul form (equivalence check)
Parent: r87a-rope2-doc-out
Type: exploit (control)
Same function as r87a (two block-diagonal dense matmuls instead of two einsums), ~0.5 ms faster. r91a/r93 used this
form and were worse; this checks whether the form itself changes training. Expect: equal to r87a mean (0.92883).
