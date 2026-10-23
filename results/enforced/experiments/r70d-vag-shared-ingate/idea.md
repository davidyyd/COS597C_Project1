# One input gate per block shared by attention and MLP
Parent: r65d-vag-attn-mlp-ingates (0.94228)
Type: explore

Separate attention and MLP input gates gave -0.0040 together on the fast base. Compute one per-channel gate from the
block input and apply it to both sublayers' (normalized) inputs: fewer gate computations; tests whether the two
selections are the same. Compare with r65d.
