# Lookahead value embeddings: keys also offer the NEXT token's embedding (exactly causal)
Parent: r18a-bf16-rope. Type: explore.

Hypothesis: an induction head needs two layers today: a previous-token head writes "my predecessor was A"
into position s+1, and a later head matches the current token A against it and copies token s+1. But for
a query at t, every key s < t has its successor token s+1 <= t already in the past. If each key s also
offers the embedding of token s+1 as an extra value, a single head can match "token A at s" and directly
retrieve "what followed A" (one-layer induction / copying), without spending a layer on it.

Change (layer 7, full attention, which has a VE table; no new tables):
  y_t = sum_{s<=t} p_ts v_s + sum_{s<t} p_ts n_s,   n_s = a_h * VE(token_{s+1}),  a_h per head init 0.
The second sum uses the same attention probabilities but excludes s = t (whose successor is the future):
computed by a strictly-causal flex pass and renormalised exactly with the two log-sum-exps,
  y = y_causal(v) + exp(lse_strict - lse_causal) * y_strict(n).
(Key 0 is kept in the strict mask only so that row 0 is never empty; row 0's term is multiplied by 0.)
Verified on CPU: perturbing token t+1 changes no output at positions <= t (max diff exactly 0).

Confirm: < 0.95739 (r18a). Refute: no gain (induction is cheap enough already at this scale).
Closest work: induction heads (Olsson et al. 2022), smeared keys (the key-side counterpart), value
embeddings. I know of no prior "successor-value" attention.
