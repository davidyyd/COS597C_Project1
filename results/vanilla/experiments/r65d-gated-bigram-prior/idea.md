# Context-gated bigram logit prior
Parent: r63d-bigram-logit-prior (folded into the head matmul). Type: explore.
The bigram prior should matter more where context is weak (document starts, after punctuation) and less
where the network is confident; a per-token gate g(x) = 2 sigmoid(w · x_final[:32]) scales it.
Confirm: < 0.954901 (r63d).
