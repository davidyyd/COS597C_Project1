# U-net skip connections (layer i output added to layer L-1-i input, learned scalar)
Parent: r25a-bf16-resid. Type: explore (known method, tested as a candidate component).

Hypothesis: later layers benefit from direct access to early-layer features (which the residual stream
overwrites); the depth readout helped for the head. Mirrored skips with learned weights (init 0) are a
cheap generalisation to all late layers.
Confirm: < 0.95726 (r25a). Closest work: modded-nanogpt U-net skips, DenseFormer.
