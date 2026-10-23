# Previous-token embedding in queries as well as keys
Parent: r03b-qtanh-prevtok
Type: exploit

Keys already carry token s-1 (r03b). Giving queries token t-1 too (the same shifted input feeds W_q and W_k: u = norm(x) + lambda ⊙ x0_{t-1}; q = W_q u, k = W_k u; one shared lambda to keep the step time)
lets a head match bigram contexts directly: "(t-1, t) == (s-1, s)" -> retrieve what followed s.
Confirm: < r03b by > 0.0007.
