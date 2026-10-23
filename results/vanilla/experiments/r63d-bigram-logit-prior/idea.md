# Low-rank bigram logit prior: a direct current-token -> next-token path into the logits
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: a large share of next-token prediction is bigram statistics, which the network must route
through the residual stream and the output head; a rank-64 factorised bigram table added to the logits
(logits += U[x_t] V^T; U random, V zero at init) learns them directly and frees the network for context.
Cost ~1% FLOPs, 1M params. Confirm: < 0.95726. Closest work: bigram/n-gram logit mixing (e.g. backoff
n-gram + neural), "skip" connections from input embedding to output. Difference: low-rank learned
bigram logits trained jointly.
