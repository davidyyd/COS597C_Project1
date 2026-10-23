# proxy (bf16 resid): value residual from layer 1 (learned scalar per layer, init 0.5)

Parent: r59a-proxy-bf16resid (0.76505). Explore: ResFormer value residual learning (and modded-nanogpt MUDD #81 'skip connections to attention values'): every later layer adds lambda * (layer-1 values) to its own values. Complements the token-indexed value embeddings with context-dependent first-layer values.
