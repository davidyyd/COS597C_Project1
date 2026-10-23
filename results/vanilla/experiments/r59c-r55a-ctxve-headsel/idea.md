# r55a + per-head selective contextual value-embedding lags
Parent: r55a-r52a-nomlpgate. Type: exploit.
Token-gated ctx-VE lags helped (r46c, -0.0006) but hit a slow kernel (+28 s); this variant gates per head
and lag (gates from x0[:32], repeated over the head's channels), a different computation pattern.
Confirm: < 0.930372 within the cap.
