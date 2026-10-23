# RoPE base proportional to receptive field in all layers
Parent: r73d-best-rope-per-window
Type: exploit

Hypothesis: r73d set each multi-scale head's RoPE base to 10000 · window/1024 (slight gain). Applying the same rule to
the full-context layers (window 2048 -> base 20000) completes it: one rule, base ∝ receptive field.
Change: full-attention layers use base 20000.
Confirm: below r73d by > 0.0005; refute: equal or worse.
