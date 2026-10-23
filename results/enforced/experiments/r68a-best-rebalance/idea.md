# Best: low-rank gate in layers 1,3,5 and MLP input gates in layers 3-7
Parent: r64a-ingate-instead (0.93169 / repeat 0.93179)
Type: exploit (time-neutral rebalance)

Move one layer's worth of gating from the head self-gate (layer 7) to the MLP input gates (layer 3).
Confirm: < r64a.
