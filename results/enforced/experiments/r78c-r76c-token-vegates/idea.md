# Token-conditioned value-embedding gates
Parent: r76c-token-headgate
Type: exploit
Hypothesis: the share of token identity (VE) vs contextual projection in each head's values plausibly depends on
the token itself (function vs content words). Add per-token biases to the projection and VE gates of the four VE
layers (8 per layer, 32 per token) from the sparse token table. (Built on r76c: on r77a it measured 69.1 ms.)
Change: g1 = 2σ(W1 x + B1[token]), g2 = 2σ(W2 x + c f_doc + B2[token]), B init 0.
Confirm: below r76c (mean 0.93079) by > 0.0003.
