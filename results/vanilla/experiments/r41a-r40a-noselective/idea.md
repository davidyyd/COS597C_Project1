# Ablation of r40a: remove the selective (token-gated) smear gate (plain query/key smears remain)
Parent: r40a-r39b-noptr-noreadout. Type: exploit (ablation).

r40a (0.932561 @ 260.4 s) uses the whole time budget. This prices the selective (token-gated) smear gate (plain query/key smears remain) inside the current best stack
(val lost per second saved), to decide where to spend the budget next (e.g. MLP gates: -0.0034 alone
but ~18 s). Note: run noise in these models is up to ~5e-4.
