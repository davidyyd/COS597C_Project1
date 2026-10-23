# Per-token output confidence: restore the magnitude the final RMSNorm discards
Parent: baseline. Type: explore.

Hypothesis: the final RMSNorm (no gain) maps the residual to a fixed norm, so the head can only express
confidence (how peaked the next-token distribution is) through the direction of x, sacrificing
directions. A per-token inverse temperature read from the residual gives an explicit confidence degree
of freedom.

Change: logits = softcap(exp(w · norm(x[:32])) * W_out norm(x)), w init 0.
Confirm: < baseline. Refute: no gain.
Closest work: learned/adaptive temperature scaling for calibration (post-hoc), nGPT logit scaling
(per-vocab, static). Difference: per-token, learned in training, before the soft-cap.
