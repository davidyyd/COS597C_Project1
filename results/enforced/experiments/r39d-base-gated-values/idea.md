# Gated normalized value mixing on a simple model
Parent: r01b-attn-selfgate
Type: explore (generality test of the value-path principle)

The value path found in the chained model (VE layers: v = s(g1 norm(W_v u) + g2 norm(ve)); other layers
v = s norm(W_v u); gates 2 sigmoid(W x) from the full input) gave ~-0.0062 on top of r28a. Test the same change
on the simple self-gate model to see whether the principle (normalize each value source, mix with token gates)
is general. Confirm: < r01b (0.95653) by > 0.002.
