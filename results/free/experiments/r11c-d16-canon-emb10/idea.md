# Embedding LR multiplier 10x (from 5x) at scale

Parent: r09a-d16w1792-canon-muonfix. Exploit (hparams): embeddings (wte, value embeds) AdamW LR 5e-2 instead of 2.5e-2; nanochat uses ~0.2-0.3 for embeddings. Group LRs were only tuned once at small scale.
