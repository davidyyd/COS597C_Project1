# d21 bigram 131k + backout + emb WD 0 + bigram LR/3 + bf16 grad all_reduce

Parent: r19b-d21-bigram-backout + bigger table (2x; 4x helped at d21) + emb WD 0 (r21d) + table LR/3 (r21c) + bf16 all_reduce. Memory ~77GB at b16 (like r19a).
