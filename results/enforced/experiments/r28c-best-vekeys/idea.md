# Best model + value embeddings also into keys (full-input gate)
Parent: r26a-vegate-full
Type: exploit

k += tanh(W' x) * ve in the VE layers (r01d: -0.0014 on baseline; no gain on the scalar-gate model r02a).
With the full-input gates of r26a, retest. Confirm: < r26a by > 0.0005.
