# Multi-scale heads in sliding-window layers
Parent: baseline
Type: explore

Hypothesis: in the short-window layers all 4 heads share a 1024-token window. Giving heads geometrically
spaced windows (128, 256, 512, 1024) forces a local-to-global division of labour (local heads cannot be
distracted by far tokens) without losing the long range of the last head; long layers stay full.

Change: sliding mask for head h: q - k <= 1024 >> (3 - h).

Confirm: < baseline by > 0.001. Refute: worse (local heads lose useful context).
Closest work: MSWA (multi-scale window attention, 2025), Longformer per-layer windows. Differs: per-head
geometric windows inside the existing SSSL pattern, nothing else changed.
