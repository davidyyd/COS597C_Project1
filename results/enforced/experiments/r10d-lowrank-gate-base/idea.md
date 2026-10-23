# Self-gate (scalar + rank-32 per-dim) alone on the baseline
Parent: baseline
Type: explore (ablation)

Measures the standalone effect of the full self-gate y_h <- y_h * 2 sigmoid(a_h . y_h + U_h V_h y_h)
(the core of the best model) on the baseline, to compare with gated attention results and to seed models
that do not use the query temperature. Confirm: < r01b (0.95653, scalar only).
