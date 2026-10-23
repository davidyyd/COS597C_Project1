# Direct low-rank bigram path to the logits
Parent: baseline
Type: explore

Hypothesis: a large share of next-token statistics is bigram-like (depends on the current token only). The
baseline must carry these through the residual stream to the final norm and head, where they compete with
contextual features. A direct low-rank path logits += V e[idx_t] (e: 8192x16 table, V: 8192x16, V init 0)
stores the bigram prior outside the residual stream, freeing it for context.
Confirm: < baseline by > 0.001. Refute: no gain (the stream carries bigram info cheaply via x0).
Closest work: n-gram / bigram heads, hashed n-gram embeddings (input side). Differs: low-rank bigram logit
term added to the head output.
