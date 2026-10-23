# Document-position logit prior (log2 bucket of tokens since BOS, rank 16)
Parent: r63d-bigram-logit-prior's head-folding on the r25a control. Type: explore.
Token distributions differ at document starts (titles, headers) vs. deep inside documents; a tiny prior
indexed by the log2 bucket of the position within the document. Confirm: < 0.95726.
