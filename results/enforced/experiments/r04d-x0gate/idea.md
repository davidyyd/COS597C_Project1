# State-dependent re-injection of the token embedding
Parent: baseline
Type: explore

Hypothesis: the baseline re-adds x0 (the token embedding) before every block with one scalar per layer.
How much a position needs to be re-grounded in its own token identity should depend on the position (e.g.
mid-word pieces vs. function words, or when context has drifted). Make the x0 weight per token:
x <- r_i x + a_i (1 + tanh(u_i . norm(x))) x0, u_i in R^512 init 0.
Confirm: < baseline by > 0.001. Refute: no gain (a per-layer scalar suffices).
Closest work: x0/value residual (ResFormer, modded-nanogpt x0 lambdas), highway gates. Differs: a bounded
per-token scalar gate on the embedding re-injection only.
