# d16 bigram 262k with 3x higher LR on the bigram table

Parent: r19c-d16-bigram262k. Explore (optimizer for sparse tables): each bigram row appears in few batches, so it gets few effective updates; try LR 3x the embedding LR (15x base) for the bigram table only.
