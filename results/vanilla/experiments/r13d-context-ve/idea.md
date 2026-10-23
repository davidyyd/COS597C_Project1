# Contextual value embeddings: causal per-channel conv over each value embedding
Parent: baseline. Type: explore.

Hypothesis: contextual x0 (a width-4 causal per-channel conv over the token embeddings) was the largest
single cheap win (-0.0057). Value embeddings inject token identity into attention values; making them
contextual (a local n-gram summary) gives the retrieved content of every attended position its local
context, e.g. copy heads retrieve "the phrase", not just the token.

Change: ve_l <- sum_{j<4} w_{l,j} ⊙ VE_l(t_{i-j}), w per table and channel, init identity.
Confirm: < baseline. Refute: no gain.
Closest work: value embeddings (modded-nanogpt), Canon layers (conv in blocks). Difference: conv on the
value-embedding lookups.
