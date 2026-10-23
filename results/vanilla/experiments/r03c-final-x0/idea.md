# Re-inject the token embedding before the output head
Parent: baseline. Type: explore.

Hypothesis: the x0 re-injection helps every block; the output head, which must output bigram-like
statistics conditioned on the current token, sees only the last residual (where the current token's
identity may have been overwritten). A learned x0 skip into the head gives it direct access.

Change: logits = W_out norm(norm(x_L) + λ x0), λ learned scalar init 0.
Confirm: < baseline. Refute: no gain (the residual already carries token identity to the end).
Closest work: DenseNet-style input skips, x0 lambdas (nanochat). Difference: skip into the head.
