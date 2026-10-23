# Queries read the token identity undiluted
Parent: baseline
Type: explore

Hypothesis: many heads search by "what token am I" (copying, induction: find earlier occurrences of the
current token). After a few layers the current token's identity is a small part of norm(x). Give only the
queries a direct copy: q = W_q(norm(x) + beta_l x0), beta init 0 (keys and values unchanged).
Confirm: < baseline by > 0.001.
Closest work: x0 re-injection (all of the stream), prev-token keys (r01a). Differs: query-only read of the
current token embedding.
