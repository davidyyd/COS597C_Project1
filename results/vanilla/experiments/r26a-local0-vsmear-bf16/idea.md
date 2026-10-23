# Local attention layers 0-1 + value smear, bf16 residual (combine the two over-cap wins)
Parent: r25d-r19a-bf16-resid (adds r19c's value smear). Type: exploit.

With the bf16 residual stream, r25d (local layer 0) and r25c (value smear) each fit with ~14 s to spare;
they act on different things (content-based local mixing in layer 0; bigram content in values).
Confirm: < 0.93626 within the cap.
