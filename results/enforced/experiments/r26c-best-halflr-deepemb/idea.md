# Best model: low-rank gate in half the layers + deep embedding
Parent: r20b-best-doctemp
Type: exploit

On the cheap lineage, using the low-rank self-gate only in odd layers kept most of its gain (r25a 0.94441)
and saves ~2 ms. Do the same on the best model and spend the time on the deep embedding (-0.0012 on r22b).
Confirm: < r20b (0.94439).
