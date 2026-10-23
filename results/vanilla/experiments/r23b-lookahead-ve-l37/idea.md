# Lookahead value embeddings in both full-attention layers (3 and 7)
Parent: r18a-bf16-rope. Type: explore.

Same mechanism as r23a, in both long-range layers (both have VE tables), to see whether one-layer
induction is useful mid-network too (so later layers can build on the copied token).
Confirm: < r23a.
