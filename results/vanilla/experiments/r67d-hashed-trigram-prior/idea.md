# Hashed trigram logit prior (table indexed by hash(previous, current token))
Parent: r63d-bigram-logit-prior. Type: explore.
The bigram table U[x_t] works as an output prior (r63d -0.0024) although a hashed bigram *input*
embedding failed (r01a). A table indexed by the hashed (x_{t-1}, x_t) pair (16384 rows, rank 64) gives a
trigram-context prior at similar parameter cost. Confirm: < 0.954901 (r63d).
