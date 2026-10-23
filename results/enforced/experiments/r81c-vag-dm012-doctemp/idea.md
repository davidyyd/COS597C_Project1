# Early-layer document masking + document-position query temperature (fast base)
Parent: r79c-vag-docmask-early
Type: explore
Question: on the best model, early doc masking gave a smaller gain (-0.0005) than on the fast base (-0.0028). One
suspect: the best model's doc-position query temperature already captures part of the document structure. Adding
the doc temperature to r79c tells whether the two overlap (small gain) or add (gain like on the baseline ~-0.002).
