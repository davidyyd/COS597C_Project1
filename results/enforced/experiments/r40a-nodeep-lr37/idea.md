# Best model: swap the deep embedding for the low-rank gate in layers 3, 7
Parent: r38a-gated-proj
Type: exploit (swap for step time)

With the gated normalized value path, is the deep embedding (~1.5 ms) still worth more than the per-dimension
low-rank self-gate in the full-attention layers (~1 ms, -0.0007 on r35a)? Confirm: < r38a (0.93717).
