# Canon computed in bf16 (speed/memory)

Parent: r09a-d16w1792-canon-muonfix. Explore/exploit (efficiency). Canon ran in fp32 on the fp32 norm output (and saved fp32 activations: +17GB at d16). Cast input and weights to bf16 first (the following linear runs in bf16 anyway). Speed+memory test vs r09a (2546s, 74GB, 0.79473).
