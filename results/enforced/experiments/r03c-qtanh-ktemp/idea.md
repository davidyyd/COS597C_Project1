# Per-key salience in addition to per-query temperature
Parent: r02c-selfgate-qtanh
Type: exploit

Hypothesis: the query temperature (sharpness per query) helped by 0.003. The key-side analogue is a per-key
salience: k_h <- norm(k_h) * (1 + tanh(v_h . x)), letting some positions (e.g. content words, delimiters) be
strongly attractive/repulsive for every query, others neutral. Logit = s_q s_k cos(q,k) sqrt(d).
Confirm: < r02c by > 0.001.
Closest work: learned QK-norm gains (static per-channel), not per-token.
