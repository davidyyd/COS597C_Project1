# Position within the document as an input feature
Parent: baseline
Type: explore

Hypothesis: rows pack several documents, each starting with BOS. RoPE only gives relative positions, so a
token does not directly know how far it is into its document — which matters a lot for prediction
(uncertainty is high right after BOS, topic is settled later). Add a scalar feature
f_t = log(1 + t - start_of_doc(t)) / log(T) along a learned direction to the input embedding
(x0 = norm(wte[idx]) + f_t w, w init 0). Causal: start_of_doc uses only tokens <= t.
Confirm: < baseline by > 0.001. Refute: no gain (attention to BOS already provides it).
Closest work: absolute position embeddings (GPT-2, per-row), document masking. Differs: one learned direction
scaled by log-distance to the document start.
