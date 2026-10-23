# Key salience: a query-independent learned logit bias per key token
Parent: baseline. Type: explore.

Hypothesis: many attention decisions are "is this key worth attending to at all" (content words, sentence
starts, BOS) independent of the query. With QK-norm, expressing such a prior costs dot-product capacity
in every query. A per-key, per-head scalar bias b_h(x_k) gives it directly (like an additive prior over
memory slots).

Change: after RoPE and QK-norm, set the last channel q[...,127] = sqrt(d), k[...,127] = w_h · x_k[:32]
(w init 0), so logit(q,k) gets + b_h(x_k). One channel of 128 (lowest RoPE frequency) is sacrificed.

Confirm: < baseline. Refute: no gain.
Closest work: attention sinks / learned sink logits, Forgetting Transformer (distance-based data-dependent
bias), stick-breaking attention. Difference: a content-based per-key additive prior, implemented free in
the QK dot product.
