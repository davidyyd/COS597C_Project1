# Low-rank skip-bigram logit prior (current and previous token)
Parent: r63d-bigram-logit-prior. Type: explore.
Adds the previous token's own factor: logits += (U[x_t] + U2[x_{t-1}]) V^T, an additive trigram-style
prior through the same output factor. Confirm: < 0.954901 (r63d).
