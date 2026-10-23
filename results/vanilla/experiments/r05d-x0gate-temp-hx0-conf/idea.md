# x0 gate + temperature + head x0 skip + per-token output confidence
Parent: r02a-x0gate-temp (adds r03c-final-x0 and r04d-logit-confidence). Type: exploit.

Hypothesis: the two free (no time cost) head-side changes (-0.0018, -0.0014 alone) should add to r02a.
Confirm: < 0.9505 at baseline speed.
