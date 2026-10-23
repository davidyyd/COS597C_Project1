# Best + per-head attention outputs normalized before the head gate
Parent: r87a-rope2-doc-out
Type: exploit
On the cheap base, normalizing each head's output (dropping the agreement magnitude) gave -0.0012 (r91d). Transfer
to the best (block-diagonal low-rank gate form for time; no per-head scale: it cost ~1 ms and c_proj/the gate can
scale). y_h <- norm(y_h) before the self-gate. Confirm: below r87a mean (0.92883) by > 0.0003.
