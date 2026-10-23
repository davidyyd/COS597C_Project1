# Token embedding as the value embedding of the layers without a VE table
Parent: r18a-bf16-rope. Type: explore.

Hypothesis: value embeddings help (token identity injected into attention values), but only alternate
layers have a table (parameter cost). The normalized input embedding x0 is a free token-identity vector
of the right width; adding it (with the usual per-head gate) to the values of the other layers gives
every layer a value-identity path without new tables. Unlike r07c (sharing the next layer's VE table,
worse), this reuses the input embedding.
Change: in layers without a VE table, v <- v + 2 sigmoid(W x[:32]) ⊙ x0 (per head).
Confirm: < 0.95739 (r18a). Refute: no gain.
