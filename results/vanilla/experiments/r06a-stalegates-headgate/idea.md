# Stale depth gates + temperature + per-head attention output gate
Parent: r05a-stale-gates (adds r05c-head-gate). Type: exploit.

Hypothesis: the per-head output gate (-0.0054 alone) lets heads abstain, a different function from the
depth gates (x0/residual mixing) and the temperature, so it should add to r05a (0.9505).
Confirm: < 0.949 within the time cap.
