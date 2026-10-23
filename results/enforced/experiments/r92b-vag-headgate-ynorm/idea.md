# Cheap base + head gates + per-head output normalization
Parent: r91c-vag-headgate
Type: exploit
Do output normalization (r91d, -0.0012) and the head self-gate (r91c, -0.0017) stack? norm(y_h) * s_h, then gate.
Confirm: below r91c (0.93837) by > 0.0005.
