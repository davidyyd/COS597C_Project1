# d16 (bigram262k, emb WD0): AdamW beta2 0.99 (from 0.95)

Parent: r21d-d16-bigram262k-embwd0. Exploit (hparam): sparse embedding rows (bigram table) get noisy second-moment estimates with beta2 0.95; 0.99 is the common default for large models.
