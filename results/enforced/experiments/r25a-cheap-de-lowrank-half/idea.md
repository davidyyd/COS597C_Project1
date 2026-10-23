# Cheap lineage with deep embedding + low-rank self-gate in half the layers
Parent: r24b-cheap-deepemb
Type: exploit

The rank-32 per-dimension self-gate in all layers puts r24b over the step cap (70.0 ms). Use it in the odd
layers only (1, 3, 5, 7: the value-embedding layers, including both full-attention layers).
Confirm: < r24b by > 0.0008.
