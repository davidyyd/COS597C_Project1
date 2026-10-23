# Document masking 0-2 + full low-rank head gate (1,3,5,7), MLP input gates only in 5-7
Parent: r80b-best-docmask01-lr357
Type: exploit
Time trade: keep all of the low-rank head gate and mask layers 0-2, paid for by dropping the MLP input gate in
layer 4. Confirm: below r80b by > 0.0003.
