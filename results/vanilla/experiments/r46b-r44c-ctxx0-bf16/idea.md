# r44c with the contextual x0 conv computed in bf16
Parent: r44c-r42b-unet. Type: exploit (cost reduction). The width-4 conv over the fp32 embedding output
and its weight-gradient reductions run in fp32; bf16 halves that traffic. Confirm: ≈ val, faster.
