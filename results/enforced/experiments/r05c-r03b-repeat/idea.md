# Repeat of r03b (noise estimate)
Parent: r03b-qtanh-prevtok
Type: control

Identical code to r03b-qtanh-prevtok (0.95282). Measures run-to-run variation of val_loss for the same
architecture (data order / init seed are fixed by the harness, so any difference is nondeterminism of
kernels), to judge which of the ~0.001 differences so far are real.
