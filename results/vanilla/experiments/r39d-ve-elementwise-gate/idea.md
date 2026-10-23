# Per-channel (instead of per-head) value-embedding gate
Parent: r25a-bf16-resid. Type: explore.
The VE gate decides per head how much token identity enters the values; per-channel gating (32 -> 512,
init neutral) lets each value feature choose. Confirm: < 0.95726.
