# Document-so-far mean embedding added to x0 (a causal topic summary)
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: a cheap running summary of the document (mean of token embeddings since the last BOS) is a
good topic prior for every layer, available through the x0 re-injection; attention must otherwise
spend capacity to average over the whole context. Computed once at the input (cumsum + gather).
Change: x0 = norm(e_t + u ⊙ mean_{s=doc_start..t} e_s), u per channel, init 0.
Confirm: < 0.95726. Refute: no gain (attention already provides it).
Closest work: continuous cache / bag-of-words context vectors (e.g. topic-RNNs), linear attention with
uniform weights. Difference: a document-reset prefix mean injected with the embedding.
