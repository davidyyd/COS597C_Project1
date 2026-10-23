# r18b-r17a-bf16 with a bf16 residual stream
Parent: r18b-r17a-bf16. Type: exploit.

Same model as r18b-r17a-bf16 with the residual stream in bf16 (see r25a): if the quality cost is small, the time saved
lets the stronger over-cap variants (value smear r19c 0.93632, local layer-0 attention r19a 0.93636) fit
the cap. Confirm: ≈ parent's val at < 261.7 s.
