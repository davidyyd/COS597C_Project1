# Block-diagonal low-rank head gate (faster form) + gate also in layer 6
Parent: r87a-rope2-doc-out
Type: exploit
The per-head low-rank self-gate computed as two block-diagonal dense matmuls (same function as the einsum form)
saves ~0.5 ms; spend it on the gate in layer 6 (each gated layer was worth ~0.0006, r90d).
Confirm: below r87a mean (0.92883) by > 0.0003.
