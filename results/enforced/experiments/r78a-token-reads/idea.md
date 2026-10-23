# Token-conditioned undiluted-read strengths
Parent: r77a-token-controls
Type: exploit
Hypothesis: the five undiluted reads per block (attention <- m_{l-1}[t], keys <- m_{l-1}[t-1], MLP <- a[t], a[t-1],
m_{l-1}[t]) have one global strength each. How much a token should rely on the previous token's or the
previous sublayer's conclusions plausibly depends on the token (e.g. continuation pieces of a word vs
punctuation). A per-token bias for each strength (5 per layer, 40 per token) from the sparse token table.
Change: b_{l,r} -> b_{l,r} + B[token, l, r], B init 0.
Confirm: below r77a (0.93068) by > 0.0003.
