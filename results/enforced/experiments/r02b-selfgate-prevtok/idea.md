# Self-gated heads + previous-token embedding in keys (all layers)
Parent: r01b-attn-selfgate
Type: exploit

Combine r01b (self-gate) with r01a (k = W_k(norm(x) + lambda ⊙ x0_{t-1})), now in all 8 layers since the
self-gate parent measures ~61 ms leaving room. Confirm: < 0.9560 (additive gains).
Closest work: key shift (modded-nanogpt), token shift (RWKV); here only keys, from the token embedding.
