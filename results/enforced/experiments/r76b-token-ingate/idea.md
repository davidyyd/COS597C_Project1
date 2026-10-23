# Token-conditioned MLP feature selection (gate features from a per-token table)
Parent: r75b-doc-ve-gate
Type: explore

Hypothesis: the MLP input gate (layers 4-7) selects features from the input itself; what to select plausibly
depends on the current token's identity, which the late residual carries only diluted. A sparse per-token table
of rank-32 gate features gives the gate direct token conditioning (self-gating + identity, not a foreign signal).
Change: u <- u * 2σ(U (D u + E_l[token])), E: 8192 x (4 x 32) table (embedding, fills the embedding cap), init 0.
Confirm: below r75b (0.93102) by > 0.0005.
Closest work: per-layer token embeddings (Gemma 3n PLE adds them to the residual), bigram/value embeddings;
here a token table that only steers a feature-selection gate.
