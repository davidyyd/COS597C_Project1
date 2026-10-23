# Hard induction pointer: hidden state of the successor of the previous occurrence
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: induction heads must learn to find the previous occurrence of x_t and read what followed it.
Compute the previous occurrence s of x_t in the same document exactly (per-row sort), take the layer-2 input
state at s+1 (<= t, causal: it encodes x_{s+1} in its context) and add P norm(state) at layer 4 (P zero-init).
Unlike r75 (successor TOKEN embedding: negative), the contextual state tells the model how similar the
earlier context was. Confirm: < 0.95726. Closest: induction heads (Olsson et al.), our r75 induction candidate.
