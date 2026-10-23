# Stretchable time: content-dependent rotary position increments
Parent: r01b-attn-selfgate (baseline + scalar head self-gate; on the plain baseline this measured 71.6 ms)
Type: explore

Hypothesis: RoPE measures distance in tokens, but tokens carry very different amounts of "content"
(subword continuations, whitespace, punctuation vs. content words). If each token advances the rotary
clock by a learned amount, relative distances become content-weighted: p_t = sum_{u<=t} (1 + tanh(w . x0_u)),
step in (0, 2), w init 0 (standard RoPE at init). Same p for q and k, so attention still depends only on the
(content-weighted) distance p_t - p_s. One 512-vector of parameters; cos/sin computed per batch.
Confirm: < r01b (0.95653) by > 0.001. Refute: no gain.
Closest work: CoPE (contextual position encoding: gated counting per query-key pair, costly), learned
position ids. Differs: one token-level step size from the token embedding, cumulative sum, used in RoPE.
