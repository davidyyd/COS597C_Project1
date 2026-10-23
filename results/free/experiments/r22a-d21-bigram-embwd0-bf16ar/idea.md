# d21 bigram 65k + backout + embedding WD 0 + flat bf16 grad all_reduce

Parent: r19b-d21-bigram-backout (0.77822) + WD 0 on embedding tables (r21d: -0.0016 at d16) + bf16 flat grad all_reduce (fixed: buffer freed before next step).
