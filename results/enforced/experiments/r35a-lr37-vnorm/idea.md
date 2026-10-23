# Low-rank gate only in the full-attention layers, plus value normalization
Parent: r28a-best-outtemp
Type: exploit

r28a has no step-time headroom (value norm on top: 70.0 ms). Keep the per-dimension low-rank self-gate only in
layers 3 and 7 (the full-attention layers; saves ~1 ms) and add per-head value normalization (r34d, -0.0016
on r01b). Confirm: < r28a (0.94335).
