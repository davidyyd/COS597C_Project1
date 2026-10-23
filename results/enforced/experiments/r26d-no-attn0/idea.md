# No attention in layer 0 when the input already has bigram features
Parent: r24d-bigram-deepemb
Type: explore

Hypothesis: layer 0's attention over raw embeddings mostly builds local (previous-token) context. With the
non-linear bigram embedding (r24d) that context is already in x0, so layer-0 attention may be redundant;
removing it saves ~1/8 of attention time, which other mechanisms need. Confirm "redundant": within +0.001
of r24d (0.95442) at lower step time.
Closest work: dropping attention layers (modded-nanogpt removed some), smear. Differs: justified by the
bigram embedding.
