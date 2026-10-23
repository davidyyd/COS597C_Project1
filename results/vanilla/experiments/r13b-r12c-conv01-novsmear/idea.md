# r12c with conv layers 0-1 and without value smear
Parent: r12c-packed-readout-ctxx0. Type: exploit.

Safer time budget version: conv in layer 1 (-13 s) and drop value smear (-10 s in r10c, -0.001).
Tests whether value smear still matters once contextual x0 provides local context.
Confirm: < 0.9425 well under the cap.
