# Contextual n-gram logit prior on the control (lags 0-3 of x0 -> rank 64 -> logits, folded into the head)
Parent: r63d-bigram-logit-prior. Type: explore.
Tests whether a direct low-rank path from the last 4 tokens' embeddings to the logits beats the plain
bigram prior (r63d -0.0024 vs r25a). Confirm: < 0.954901.
