# r25a + bf16-state AdamW + bigram-hashed value embedding on the last layer

Parent: r25a-d21-262k-b16-mu09 (best, 0.77503) + bigram-hashed last-layer VE (r25d: -0.0008 at d16). r25a peaks at 79.0GB, so AdamW states go to bf16 (custom AdamWBF16, saves ~2.6GB) to make room for the +103M-param VE table.
