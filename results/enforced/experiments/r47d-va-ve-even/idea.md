# Value embeddings on the even layers
Parent: r43a-base-valuepath
Type: explore

With normalized value embeddings their placement may matter differently: the baseline puts them on layers
1, 3, 5, 7 (always including the last). Test 0, 2, 4, 6 (the first layer gets token identity in its values,
the last does not). Confirm/refute: compare with r43a (0.94690).
