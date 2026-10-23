# r12c with layer 1 also a value-convolution layer
Parent: r12c-packed-readout-ctxx0. Type: exploit.

r12c is the best model (0.9413) but runs 274 s. Replacing layer-1 attention by a value conv saved 13 s
for +0.0008 in r10b. With contextual x0 already providing local context, layer 1's attention may matter
even less. Confirm: ≈ 0.9413-0.942 at ≤ 261.7 s.
