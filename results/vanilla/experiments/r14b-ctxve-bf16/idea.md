# Contextual value embeddings in bf16 (speed/quality check of r13d)
Parent: r13d-context-ve (same idea, bf16 implementation). Type: exploit.

Same model as r13d except the value embeddings and the conv run in bf16 (the attention value path is
bf16 under autocast). Measures how much of r13d's +44 s was fp32 traffic, and whether the gain holds.
Confirm: ≈ 0.950 at much less than 294 s.
