# Layer-1 value embedding indexed by the previous token
Parent: r73d-best-rope-per-window
Type: explore

Hypothesis: value embeddings give attention token identity at the source. In layer 1, looking the value
embedding up by the PREVIOUS token makes each source position carry the identity of the token before it, so a
query can retrieve "what preceded this token" (a bigram feature at the source) without having to build it.
Change: layer 1: ve = VE_1[idx[t-1]] (other VE layers unchanged). No new parameters, no extra compute.
Confirm: below r73d by > 0.0005.
Closest work: bigram hash embeddings (modded-nanogpt), smear/token shift; here a shifted VE lookup in one layer.
