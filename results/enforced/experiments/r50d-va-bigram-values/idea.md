# Bigram-hashed value embeddings as a third normalized value source
Parent: r43a-base-valuepath
Type: explore

Hypothesis: with normalized gated mixing, value sources combine safely. The previous-token VE row as a third
source gave little (r41d), but a dedicated table indexed by the hashed (previous, current) token pair carries
bigram identity (what phrase is at the attended position). 8192x128 table (spare embedding budget), normalized
at use, broadcast to all heads, signed gate (0 at init), in VE layers. Confirm: < r43a (0.94690) by > 0.001.
Closest work: hashed bigram embeddings (input side), value embeddings. Differs: bigram table in the values.
