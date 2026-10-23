# Head output = normalized retrieval x self-gate
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: value normalization helped a lot (r35a); the attention output then has a direction (what was
retrieved) and a magnitude (how much the attended values agree). Decouple them explicitly: the head writes
norm(y_h) scaled by the self-gate 2 sigmoid(a_h . y_h), where the gate is computed from the unnormalized y_h
(so it can use the agreement). The magnitude of the write is then entirely the gate's decision.
Confirm: < r01b (0.95653) by > 0.001.
Closest work: gated attention, nGPT-style normalization. Differs: per-head normalized retrieval with a
self-gate deciding the write magnitude.
