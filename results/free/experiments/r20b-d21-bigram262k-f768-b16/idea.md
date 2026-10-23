# d21 bigram 262139 x 768 factorized (+ up-projection) + backout, batch 16

Parent: r19b-d21-bigram-backout. Exploit: get the bigger table at batch 16 by factorizing (262139 x 768 embedding, normal init, then zero-init Linear 768->1792); memory ~ the same as the 65521 x 1792 table.
