# Row-position attention temperature (ablation of the document-relative length)
Parent: r90d-doclen-temperature. Type: explore (controlled ablation). Same as r90d but with the absolute
position in the packed row instead of tokens since BOS (Scalable-Softmax-like with a learned per-head slope).
If r90d's gain comes from document-relative length, this should be clearly worse than 0.954848.
