# Mid-depth renormalization of the residual stream
Parent: baseline
Type: explore

Hypothesis: the undiluted-read results show later sublayers are hampered because the pre-norm residual sum
keeps growing, so each new write is a shrinking fraction of what the next sublayer reads. A parameter-free
reset — x <- norm(x) once at the middle (before block 4) — restores the relative weight of the second
half's writes without changing what any sublayer can represent.
Confirm: < baseline by > 0.001. Refute: worse (norm growth is useful / harmless).
Closest work: post-norm / sandwich norm (every sublayer), nGPT (normalize after every update), Attention
Residuals (Kimi 2026). Differs: one parameter-free renormalization at mid-depth.
