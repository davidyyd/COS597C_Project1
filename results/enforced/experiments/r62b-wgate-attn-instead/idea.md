# Write gates on the attention sublayer instead of the MLP (layers 4-7)
Parent: r61a-lrodd-outgate-late (0.93223)
Type: exploit (time-neutral comparison)

Same per-channel input-driven write gate, applied to the attention write (a <- a * 2 sigmoid(U V u)) instead of
the MLP write, in layers 4-7. Tells which sublayer benefits. Confirm/refute vs r61a.
