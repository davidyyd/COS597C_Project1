# Self-gated heads + value embeddings into keys
Parent: r01b-attn-selfgate
Type: exploit

Combine the two best round-1 ideas: self-gate on each head's output (r01b, 0.95653) and the value-embedding
table also added to keys with a tanh gate (r01d, 0.95759). They act on different parts of attention
(output vs. matching) so gains should roughly add. Confirm: < 0.9560.
Closest work: gated attention, value embeddings (ResFormer); see parents' idea.md.
