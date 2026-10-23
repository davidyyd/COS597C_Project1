# Content-dependent RoPE position offsets
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: RoPE measures distance in tokens, but linguistically relevant distance is not uniform in BPE
pieces (a long word split into 5 pieces vs. a single-piece word). Let each token shift its own RoPE position
by a learned, per-head, content-dependent offset delta_h(x_t) (from 32 channels, init 0): queries and keys
are rotated as if at position t + delta(x_t), so relative positions become (t - s) + (delta_t - delta_s) and
heads can e.g. compress word pieces or punctuation in "position space".
Confirm: < 0.95726. Closest work: CoPE (contextual position encoding, Golovneva et al. 2024; counts gated
tokens), learned position offsets. Difference: a per-token continuous offset inside RoPE (cheap).
