# One more late value-embedding layer (layer 6 shares layer 7's table)
Parent: r43a-base-valuepath
Type: explore

If late layers benefit most from token identity in values, add it to layer 6 at no parameter or lookup cost
(reuses layer 7's lookup, own gates). Confirm: < r43a.
