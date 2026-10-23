# Per-token value strength
Parent: r05b-best-x0gate
Type: exploit

The query temperature (per-token sharpness) helped; per-key salience hurt. A third per-token magnitude in
attention is the value strength: v_h <- v_h * (1 + tanh(w_h . norm(x))) — how much a position contributes
when it is attended to (e.g. punctuation retrieved but with little to say), without changing where attention
goes (unlike key salience). Confirm: < r05b (0.95219) by > 0.0005.
Closest work: gated attention gates outputs at the query side; this gates at the source.
