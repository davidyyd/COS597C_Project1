# Value embeddings in every layer (pair-shared tables) with gated normalized mixing
Parent: r38a-gated-proj
Type: exploit (output temperature removed for step time)

Value embeddings in all layers with pair-shared tables hurt the baseline (r06d) when mixed unnormalized. With
gated normalized mixing (v = s(g1 norm(W_v u) + g2 norm(ve)); r37c/r38a) the balance is controlled, so the even
layers can also get token identity in their values at no parameter or lookup cost (layer 2i reuses layer
2i+1's lookup). Confirm: < r38a (0.93717).
