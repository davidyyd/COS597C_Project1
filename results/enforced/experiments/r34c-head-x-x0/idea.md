# Head reads a token x context product feature
Parent: baseline
Type: explore

Hypothesis: next-token prediction depends on interactions between the current token and the context (e.g. the
same context predicts different continuations after "the" vs "of"); a linear head over the final state must
have these interactions computed upstream. Add a cheap multiplicative feature to the head input:
x_head = norm(x_L) + beta norm(x_L * x0) (beta init 0).
Confirm: < baseline by > 0.001.
Closest work: bilinear / factorization-machine features, gated heads. Differs: elementwise product of final
state and token embedding, read only by the head.
