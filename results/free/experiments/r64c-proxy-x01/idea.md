# proxy: extra token-embedding table mixed into every layer (x01)

Parent: r59a. modded-nanogpt medium record 'decoupling embedding size from model dimension': a second independent token embedding table, normalized, added at every layer with its own learned per-layer lambda (init 0.1). Embedding capacity for ~no FLOPs.
