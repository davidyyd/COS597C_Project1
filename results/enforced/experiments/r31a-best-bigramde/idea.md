# Best model with the non-linear bigram embedding
Parent: r28a-best-outtemp
Type: exploit

The deep embedding matters in r28a (removing it costs +0.002, r30a). Its bigram version (MLP over
[e_t; e_{t-1}], hidden 256) beat the unigram one on baseline (r24d vs r23d). Swap it in. Confirm: < r28a.
