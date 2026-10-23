# Combine token x0 gate + per-head attention temperature
Parent: r01c-x0-token-gate (adds r01d-head-temperature). Type: exploit.

Hypothesis: the two round-1 wins act on different parts (embedding re-injection vs. attention sharpness),
so they should add (~-0.008 vs baseline).

Change: r01c + q <- exp(s_h) norm(q).

Confirm: below both parents. Refute: no additivity (they fix the same weakness).
Closest work: as in the parents.
