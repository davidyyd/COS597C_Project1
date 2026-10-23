# Value embeddings in every layer by sharing tables between layer pairs
Parent: baseline. Type: explore.

Hypothesis: value embeddings (token identity injected into attention values) help, but only alternate
layers have them because of their parameter cost. Sharing each table with the preceding layer (with its
own per-head gate) gives every layer a token-identity value path at no parameter cost.

Change: layer 2m uses value_embeds[2m+1](idx) with its own ve_gate; tables unchanged.
Confirm: < baseline. Refute: no gain (alternate layers are enough / sharing conflicts).
Closest work: value residual learning (ResFormer), modded-nanogpt value embeddings (shared across
symmetric layers). Difference: adjacent-layer sharing to cover all layers, own gates.
