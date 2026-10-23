# proxy backin5-mix + learned output logit bias (unigram prior)

Parent: r67a-proxy-backin5-mix (0.76388). The untied lm_head has no bias; add a learned [vocab] bias before the softcap (zero init, AdamW head group LR).
