# Best + per-head output normalization (with per-head scale) in early layers 0-3
Parent: r87a-rope2-doc-out
Type: exploit
Output normalization with a learned per-head scale gave -0.0012 on the cheap base (r91d) and stacks with the head
gate (r92b, -0.001); without the scale it hurts (r92a/r92c). In all layers it exceeds the time cap (69.2 ms), so
only early layers 0-3 (block-diagonal low-rank gate form for time). Confirm: below r87a mean (0.92883).
