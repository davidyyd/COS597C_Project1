# d16 (r30d): sign trick on the hashed bigram embedding

Parent: r30d-d16-nosmear (0.78214). From modded-nanogpt record 83 (hashing trick, Chen et al. 2015): each bigram gets a fixed pseudo-random +-1 vector (from two extra hashes, XOR of 31- and 29-periodic bit patterns over channels) multiplied onto its (shared) table row, so colliding bigrams interfere less.
