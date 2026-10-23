# Swap the half-layer low-rank output gate for a per-dimension query gate
Parent: r28a-best-outtemp
Type: exploit

Two per-dimension gates helped in other lineages: on the attention output (low-rank, r08b) and on the query
(rank 16, r10b). Time allows only one; test the query gate in all layers instead of the output gate in half.
Confirm: < r28a (0.94335).
