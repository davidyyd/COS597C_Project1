# bf16 residual stream (control on r18a)
Parent: r18a-bf16-rope. Type: explore (precision/efficiency control, disclosed as such).

Profiling shows most of the non-matmul time is memory traffic of elementwise ops on the residual stream,
which is fp32 only because the embedding output is fp32 (every other activation is bf16 under autocast).
Casting the normalized embedding to bf16 makes the residual stream bf16. This measures the quality cost
and the speed gain; not an architectural idea by itself.
Confirm: val within ~0.001 of r18a (0.95739) and clearly faster.
