# RoPE on the value path: values carry their relative offset
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: attention outputs are position-blind mixtures of values: a head cannot report *where* (how
far back) the attended content was, only what it was. Rotating each value by its key position and the
output back by the query position, y_t = R(-t) sum_s p_ts R(s) v_s = sum_s p_ts R(s - t) v_s, encodes the
relative offset of the attended tokens in the output (cheap: two rotations).
Confirm: < 0.95726 (r25a). Refute: no gain. Closest work: RoPE (on q/k only); "VO-RoPE"-style proposals
for values (2025 blog/papers). Difference: tested in a small LM with QK-norm and value embeddings.
