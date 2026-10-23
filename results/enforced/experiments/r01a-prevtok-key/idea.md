# Previous-token embedding injected into keys
Parent: baseline
Type: explore

Hypothesis: induction / copy heads need keys that encode "the token before me". In the baseline the first
layers must spend attention to build this. Feeding the (normalized) embedding of token t-1 directly into the
key projection of layers 0,2,4,6 gives them a free previous-token key feature.

Change: k_t = W_k (norm(x_t) + lambda_l ⊙ x0_{t-1}), lambda_l per-channel per layer, init 0 (AdamW).
Queries and values unchanged.

Confirm: val_loss below baseline (0.95895) by > ~0.002. Refute: no change.
Closest work: key-shift in modded-nanogpt (shift half the key dims from position t-1 of the same layer),
RWKV token shift, "smear" of embeddings. Differs: only keys, and the shifted signal is the raw token
embedding x0 (pure token identity) rather than the layer's own state, added in the 4 layers without value embeddings (0,2,4,6).
