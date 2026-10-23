# Head gate + per-head output normalization in early layers only (cheap base)
Parent: r92b-vag-headgate-ynorm
Type: explore
Early-only normalization beat all-layer normalization without head gates (r94d 0.93831 vs r91d 0.93883). With the head
gate: compare with r92b (all layers, 0.93737) and r91c (none, 0.93837).
