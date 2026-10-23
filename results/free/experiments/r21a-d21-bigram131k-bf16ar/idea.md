# d21 bigram 131071 rows + backout, b16, flat bf16 grad all_reduce

Parent: r19b-d21-bigram-backout (0.77822) + table 2x (r20a: 4x table gave -0.0014 at d21 but needed b8 and 3568s). 131071 rows adds the same memory as r19a's trigram (76.9GB, fit at b16). Plus one flat bf16 grad all_reduce (speed).
