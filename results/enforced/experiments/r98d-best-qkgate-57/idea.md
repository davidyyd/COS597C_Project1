# Final model + q/k input self-gate in layers 5 and 7
Parent: r94a-best-blockdiag
Type: exploit
The q/k-only input self-gate (feature selection for matching) gave -0.0018 on the fast base (r67a) but never fit
on the best model; the block-diagonal gate form freed ~1 ms. Add it in layers 5 and 7.
Confirm: below r94a mean (0.92887) by > 0.0003.
