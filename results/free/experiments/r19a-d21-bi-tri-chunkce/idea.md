# d21 bigram + trigram hash embeddings (65521 each) + chunked softcap CE (memory)

Parent: r18a-d21-bigram (best, 0.77926). Exploit/explore: add a hashed trigram embedding too; to fit the extra 1.9GB at batch 16, the lm_head+softcap+CE is a custom autograd function over 4 row chunks that recomputes logits in backward (saves ~3-4GB of fp32 logits).
