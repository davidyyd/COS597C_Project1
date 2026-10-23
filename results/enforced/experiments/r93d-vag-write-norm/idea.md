# Attention write normalized (unit vector x learned per-layer scale), on top of per-head output normalization
Parent: r91d-vag-yhead-norm
Type: explore
Hypothesis: head-output normalization helped (r91d); going one step further, the attention write to the residual as a
unit vector with a learned per-layer scale (init 0.1; c_proj starts at zero) fixes each layer's contribution size
per token. Per-token write multipliers hurt before, but this removes magnitude rather than adding a multiplier.
Confirm: below r91d (0.93883) by > 0.0005.
