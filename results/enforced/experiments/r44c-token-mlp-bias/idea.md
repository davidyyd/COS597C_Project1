# Token-conditioned MLP pre-activation bias from a normalized sparse table
Parent: r43a-base-valuepath
Type: explore

Hypothesis: the value-embedding fix showed token-specific tables are powerful once normalized at use. Give the
MLPs a token-specific signal at the neuron level: hb_t = U norm(E[idx_t]) (E: 8192x128 embedding table using the
spare embedding budget, U: 128x2048 shared), added to every layer's MLP pre-activations with a per-layer scalar
(init 0): relu(W1 u + beta_l hb_t)^2, so which hidden units fire can depend on token identity directly.
Confirm: < r43a (0.94690) by > 0.001.
Closest work: value embeddings (attention values), hashed n-gram embeddings (input). Differs: token-specific
biases on MLP hidden units, low-rank, normalized table.
