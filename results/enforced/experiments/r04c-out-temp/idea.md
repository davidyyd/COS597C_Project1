# Per-token output temperature (confidence) after the gainless final norm
Parent: baseline
Type: explore

Hypothesis: RMSNorm without gains before the LM head erases the residual's magnitude, so the logit scale of
a token can only change through the direction of x relative to lm_head rows. Confident vs. uncertain
positions need very different logit scales. The q-temperature result (r02c, -0.003) suggests restoring a
bounded magnitude channel where a gainless norm removed it helps. Here at the output.

Change: logits = lm_head(norm(x)) * (1 + tanh(w . norm(x))), w in R^512 init 0 (then soft-cap 15 as before).

Confirm: < baseline by > 0.001. Refute: no gain (the head already handles scale via direction).
Closest work: temperature scaling / learned temperature in calibration (post-hoc, global); here per token,
trained end to end.
