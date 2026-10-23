# Cross-layer KV sharing between adjacent sliding-window layers (2 <- 1, 6 <- 5)
Parent: r25a-bf16-resid. Type: explore (known method, as a possible time saver).
Hypothesis: adjacent sliding-window layers compute similar keys/values; letting layers 2 and 6 reuse the
(rotated, normalised) keys and values of layers 1 and 5 drops two k/v projections, RoPE and norms per shared
layer with little quality loss, freeing time for components that are over the cap.
Confirm: ≈ 0.95726 and faster. Closest work: Cross-Layer Attention (Brandon et al. 2024), YOCO.
