# Local repetition pattern embedding
Parent: r25a-bf16-resid. Type: explore.
Exact document-repetition features help (r85d-r87c). Test the purely local version: a 4-bit pattern
[x_t == x_{t-k}] for k = 1..4 indexes a learned embedding (16x512, zero-init) added to the input.
It marks reduplication, list/table structure and loops without attention. Confirm: < 0.95726.
Closest: our occurrence counts (document-level); RWKV/conv token shift (no equality features).
