# Learnable per-head attention temperature (q gain, init 1 on top of scale 0.12)

Parent: r10d-polar-express. Explore (attention). Gain-free QK-norm fixes the logit range per head; a learnable per-head gain on q lets heads choose sharpness (attn scale 0.12 helped, suggesting the fixed range is limiting).
