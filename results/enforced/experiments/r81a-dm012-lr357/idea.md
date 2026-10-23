# Best + document masking in early local layers 0-2, low-rank head gate in 3, 5, 7
Parent: r80b-best-docmask01-lr357
Type: exploit
Fast base: masking layers 0-2 (-0.0028) beats 0-1 (-0.0017). r80b masked 0-1 (0.93019); extend to layer 2 with the
same low-rank layers (3, 5, 7). Confirm: below r80b by > 0.0003.
