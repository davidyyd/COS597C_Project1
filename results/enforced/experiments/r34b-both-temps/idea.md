# Per-token temperature both before and after the soft-cap
Parent: r28a-best-outtemp
Type: exploit

Keep the pre-cap temperature (sharpness within the cap) and add a post-cap one (level of the cap per token):
logits = 15 (1 + tanh(v . x)) tanh(l (1 + tanh(w . x)) / 15). Confirm: < r28a.
