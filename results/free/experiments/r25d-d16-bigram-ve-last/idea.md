# d16 (mu 0.9): last layer's value embedding indexed by bigram hash (65521 rows)

Parent: r24c-d16-muonmu09 (0.78558). Explore: value embeddings (ResFormer) are per-token tables; the input bigram embedding helped a lot, so give the last layer's value embedding bigram context too (table indexed by hash(prev, cur) mod 65521, uniform init like the other VEs).
