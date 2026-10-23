# Keys also read the previous MLP output at t-2 (cheap lineage)
Parent: r54c-r51d-k-shift-only (0.93612)
Type: exploit

Keys get a two-token local context (MLP output at t-1 and t-2); on the best model it measured 69.3 ms.
Confirm: < r54c.
