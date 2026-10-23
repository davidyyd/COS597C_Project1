# Cheap base: early doc masking + doc temp + MLP input gates in layers 4-7
Parent: r81c-vag-dm012-doctemp
Type: exploit (stronger cheap base for explores)
Adds the MLP input self-gate (late layers, as in the best model) to the cheap doc-aware base, to get a stronger
~61 ms base for explores whose cost would not fit the best model. Confirm: below r81c by > 0.001.
