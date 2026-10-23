# Keys carry the document position (normalized bucket table)
Parent: r43a-base-valuepath
Type: explore

As r47b but in the keys (before RoPE / QK-norm): k <- norm(k) + tanh(W x) norm(PK[bucket]), so heads can attend
by document position (e.g. to the document start, or away from the previous document).
Confirm: < r43a by > 0.001.
