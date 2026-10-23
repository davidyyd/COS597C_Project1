# Value embeddings in every layer, shared by layer pairs
Parent: baseline
Type: explore

Hypothesis: value embeddings help because values get direct token identity. Only alternate layers have
them. With the same 4 tables shared by layer pairs (0,1), (2,3), (4,5), (6,7) — each layer with its own
gate — every layer gets token identity in its values at no parameter cost.
Confirm: < baseline by > 0.001. Refute: no gain.
Closest work: value embeddings (modded-nanogpt shares tables between early and late layers in a U-shape).
Differs: adjacent-pair sharing so every layer has one.
