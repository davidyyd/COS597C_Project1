# Replace the low-rank gate by value normalization
Parent: r28a-best-outtemp
Type: exploit

Drop the half-layer low-rank self-gate (~2 ms, ~-0.0011) and add per-head value normalization (r34d, -0.0016
on r01b). Compare with r35a (low-rank kept in layers 3, 7). Confirm: < r28a.
