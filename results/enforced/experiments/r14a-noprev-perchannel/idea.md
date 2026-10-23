# Three-read chain with per-channel weights (no prev-token keys)
Parent: r14b-chain3-noprev
Type: exploit

Per-channel read weights (gave -0.0009 on r12b) on the three-read chain; built on the no-prev-token-keys
variant because the full model measured 69.2 ms with them. Confirm: < r14b.
