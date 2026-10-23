# Bigram-conditioned MLP bias (hashed (prev, cur) sparse table, normalized)
Parent: r43a-base-valuepath
Type: explore

Same as r45c but the sparse table is indexed by a hash of the (previous, current) token pair (8192 buckets):
bigram-specific neuron biases hb = U norm(E[hash(t-1, t)]) in the MLPs of layers 0-3. A bigram is far more
predictive than a unigram and the table rows are normalized at use (sparse rows).
Confirm: < r45c.
Closest work: hashed n-gram embeddings (input side, e.g. modded-nanogpt bigram hash), value embeddings.
Differs: neuron-level MLP biases, normalized at use.
