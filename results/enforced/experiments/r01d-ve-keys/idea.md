# Value embeddings also feed the keys
Parent: baseline
Type: explore

Hypothesis: value embeddings give attention outputs direct access to token identity, but keys still have to
recover token identity through the residual stream, which is diluted by context after a few layers. Adding
the same per-token table to the keys (own gate, tanh so it can be signed and starts at 0) lets heads in
layers 1,3,5,7 match on exact token identity (copying, repetition) cheaply.

Change: in VE layers, k <- k + tanh(W_k' x[:32]) * ve (per head), W_k' init 0; values unchanged.

Confirm: val_loss < baseline by > 0.002. Refute: no gain.
Closest work: value embeddings / ResFormer (values only). Differs: token-identity embeddings injected into
keys before RoPE+QK-norm, sharing the value-embedding table.
