# Document occurrence count + log-distance to the previous occurrence
Parent: r25a-bf16-resid. Type: explore (refine of r85d).
Adds a second embedding of floor(log2(distance to the most recent earlier occurrence of x_t in the doc))
(0 = never seen): recency of a repetition matters (burstiness). Confirm: < 0.955723.
