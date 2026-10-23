# Multi-resolution heads in every layer (also the full-attention layers)
Parent: r26d-multires-heads. Type: explore.

r26d gave 2 local heads to each sliding-window layer (-0.002, faster). Tests whether the two full-context
layers also benefit from having 2 local heads and 2 full heads (all layers via flex with per-head masks).
Confirm: < 0.955269 (r26d). Refute: worse => the global layers need all heads global.
