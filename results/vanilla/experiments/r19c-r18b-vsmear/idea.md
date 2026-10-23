# r18b + value smear (now cheap with a bf16 value path)
Parent: r18b-r17a-bf16. Type: exploit.

Value smear gave -0.001 in r10c but cost 10 s when the value path was fp32; with bf16 values it may be
cheap now. Confirm: < 0.9369 within the cap.
