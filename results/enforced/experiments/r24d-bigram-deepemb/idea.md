# Non-linear bigram embedding
Parent: baseline
Type: explore

Hypothesis: the deep-embedding result (r23d, -0.0039) shows the re-injected x0 benefits from non-linear token
features. Many useful input features are bigram-level (word pieces, phrases). Let the refinement MLP see the
previous token too: x0 = norm(e_t + W2 relu(W1 [e_t; e_{t-1}])^2) (hidden 256, W2 init 0).
Compare with r23d (unigram MLP). Confirm: < r23d (0.95503).
Closest work: smear (mixes e_{t-1} into e_t linearly), hashed bigram embeddings (table lookup). Differs: a
small non-linear MLP over the two embeddings, whose output is the x0 re-injected at every layer.
