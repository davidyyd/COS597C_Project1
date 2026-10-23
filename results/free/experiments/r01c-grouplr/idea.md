# Per-group AdamW learning rates (embeddings 5x, head 0.4x)

Parent: baseline. Explore (optimizer). nanochat uses a much higher LR for the embeddings (~0.2-0.3) and lower for the unembedding (0.004). Here: embedding (wte, value embeds) AdamW lr 5e-2, lm_head 4e-3, Muon and scalars unchanged at 1e-2.
