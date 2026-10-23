# Document-age feature in the embedding
Parent: baseline. Type: explore.

Hypothesis: rows pack several documents; a token's prediction difficulty and the useful attention range
depend on how much of its own document precedes it. RoPE gives only relative positions, so the model
must infer "tokens since BOS" by attention. Adding a learned direction scaled by log(1 + tokens since
the last BOS)/log(T) to the embedding gives it directly (and, through x0 re-injection, to every layer).

Change: x0 = norm(wte(t) + a(t) u), a(t) = log1p(t - last_bos(t)) / log T, u learned (init 0).
Confirm: < baseline. Refute: no gain.
Closest work: absolute position embeddings (GPT-2), document-aware position resets. Difference: one
learned direction for log document age, on top of RoPE.
