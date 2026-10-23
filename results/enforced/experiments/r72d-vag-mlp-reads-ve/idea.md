# MLP reads the token's value-embedding row (undiluted token identity for the MLP)
Parent: r58b-va-gates-adam
Type: explore

Hypothesis: the value embeddings give attention an undiluted, per-layer token-identity signal, and normalizing them
was the biggest gain of this search. The MLP, which stores most per-token knowledge, sees the token only through
the diluted residual. Letting the MLP in VE layers read the same normalized table row (no new table, one scalar)
gives it a direct key into token-specific facts.

Change (VE layers 1,3,5,7): MLP input u = norm(x) + c_l · norm(VE_l[token]), c_l scalar init 0 (AdamW).

Confirm: below r58b (0.94618) by > 0.0005. Refute: no gain (the residual already carries enough token identity,
e.g. via the x0 re-add).

Closest work: value embeddings (modded-nanogpt), per-layer token embeddings into the MLP (e.g. Gemma 3n
per-layer embeddings, which use separate tables). Here: re-use the attention's VE table, normalized, as an
undiluted MLP read, with no extra embedding parameters.
