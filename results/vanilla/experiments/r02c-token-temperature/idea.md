# Token-dependent attention temperature
Parent: r01d-head-temperature. Type: exploit.

Hypothesis: r01d showed per-head temperature helps (QK-norm caps sharpness). How sharply a query should
attend depends on the token (e.g. a copy target vs. a function word), so make it per query token.

Change: q <- exp(s_h + w_h · x[:32]) norm(q), w_h init 0 (x is the normed block input).

Confirm: below r01d. Refute: no gain over static temperature.
Closest work: learned temperature/QK gains (OLMo2, nGPT), selective/scalable softmax (log-n scaling).
Difference: per-query learned temperature from a tiny linear read of the residual.
