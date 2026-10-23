# proxy backin5-mix + layers 1-3 attention read mix(x0, x) (learnable, init pure x0)

Parent: r67a-proxy-backin5-mix (0.76388). Mirror of the late trick for early layers: attention of layers 1-3 reads x0 + m*(x - x0) (m learnable per block, init 0), i.e. starts attending over the clean (bigram-enriched) embeddings.
