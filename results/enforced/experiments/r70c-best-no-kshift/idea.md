# Best without the key time-shifted read (ablation)
Parent: r64a-ingate-instead (0.93169 / 0.93179)
Type: exploit (simplification ablation)

Simplification pass: is this component still contributing in the final model? If removing it costs < 0.0005, the
simpler model is preferred (and frees step time). Confirm "needed": > r64a + 0.0005.
