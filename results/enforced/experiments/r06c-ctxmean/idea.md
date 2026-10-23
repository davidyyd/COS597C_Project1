# Causal bag-of-tokens context channel
Parent: baseline
Type: explore

Hypothesis: topic / document-level information must be gathered by attention, which in an 8-layer model
competes with local syntax. A free, parameter-less summary c_t = norm(sum_{s<=t} x0_s) (the normalized
running mean of token embeddings) re-injected like x0 gives every layer a global context vector.

Change: before every block x <- r_i x + a_i x0 + b_i c, b_i init 0 (scalars, AdamW).
Confirm: < baseline by > 0.001. Refute: no gain (attention already does this cheaply).
Closest work: x0 re-injection (modded-nanogpt), linear attention / mean pooling, CBOW. Differs: an explicit
causal mean of embeddings fed into the residual stream of a softmax transformer, 8 scalars.
