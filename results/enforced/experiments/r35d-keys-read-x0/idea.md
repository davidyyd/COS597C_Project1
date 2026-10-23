# Keys read the token identity undiluted
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: queries reading x0 helped (r17b) and prev-token keys helped; keys that carry the current token's
identity undiluted let heads find earlier occurrences of a token directly (copy / repetition):
k = W_k(norm(x) + beta_l x0), beta init 0. Confirm: < r01b (0.95653) by > 0.001.
Closest work: value embeddings into keys (r01d, table), prev-token keys (r01a). Differs: current-token x0 read
by the key projection (no new table).
