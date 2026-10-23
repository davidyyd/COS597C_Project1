# Decomposed trigram logit prior: current, previous and two-back token tables (rank 48 each)
Parent: r68d-skipbigram-separate. Type: explore.
Extends the additive n-gram prior to x_(t-2) with its own factors (same total parameters as r68d's
2 x rank 64). Confirm: < 0.953907 (r68d).
