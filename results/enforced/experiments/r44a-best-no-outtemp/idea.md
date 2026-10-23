# Best model without the output temperature (ablation)
Parent: r41b-best-tokgated-reads
Type: exploit (ablation / time)

Is the per-token output temperature still useful now that the value path is fixed? Frees ~0.9 ms.
Confirm "redundant": within +0.0005 of r41b (0.93659).
