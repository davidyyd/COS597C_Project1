# d21 bigram 262k + backout + emb WD 0 + bf16 all_reduce, device batch 8

Parent: r22a-d21-bigram-embwd0-bf16ar (0.77694) with the 4x table of r20a (0.77679 but 3568s total). The bf16 flat all_reduce should save ~100s on r20a's time (its 470M-param table made the fp32 all_reduce slow). Expected ~3470s total.
