# Cheap base + head self-gate (scalar + block-diagonal low-rank) in all layers
Parent: r83a-vag-dm-ingate
Type: exploit (stronger cheap base)
Ports the best model's head self-gate (worth ~0.0006 per layer there) in its faster block-diagonal form to the cheap
base, giving a stronger base for explores. Confirm: below r83a by > 0.002.
