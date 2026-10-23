# Per-head attention outputs normalized (agreement magnitude removed)
Parent: r83a-vag-dm-ingate
Type: explore (mechanism)
Value normalization was the largest gain: output magnitude then equals agreement among retrieved values. Is that
agreement magnitude useful? Normalize each head's output y_h to unit RMS (learned per-head scale).
Refute (magnitude useful): worse than r83a (0.94006).
