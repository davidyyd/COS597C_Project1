# Best model without prev-token keys
Parent: r13c-mlp-reads-prevmlp
Type: exploit (ablation that frees ~1.4 ms)

Does the induction shortcut (k input += lambda x0_{t-1}) still matter once sublayers chain their outputs
directly? Removing it frees step time for further additions. Confirm "redundant": within +0.0005 of r13c.
