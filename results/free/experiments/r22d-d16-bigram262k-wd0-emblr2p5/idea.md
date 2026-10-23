# d16 bigram 262k + emb WD 0 + all embedding LRs halved (mult 2.5)

Parent: r19c-d16-bigram262k + emb WD 0. Explore: the lower bigram LR helped; check whether all embedding tables (wte, value embeds, bigram) prefer a lower LR (mult 5 -> 2.5) once WD is off.
