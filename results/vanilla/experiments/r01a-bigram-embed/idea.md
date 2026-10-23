# Hashed bigram input embedding (trading one value-embedding slot)
Parent: baseline. Type: explore.

Hypothesis: the first layers spend attention + MLP capacity just to form the (previous, current) token
bigram feature. A direct hashed bigram lookup in the input embedding gives every layer (via the x0
re-injection) the bigram identity for free, freeing the early layers.

Change: x0 = norm(wte[t_i] + P · norm(E1[h1(t_{i-1},t_i)] + E2[h2(t_{i-1},t_i)])), two hash tables of
16384x128 and a 128->512 projection. Paid for by removing layer 1's value embedding (same param count).

Confirm: val_bpb < baseline by > ~0.003. Refute: equal or worse (bigram info is cheap to compute anyway).

Closest work: hashed n-gram embeddings (Engram/DeepSeek 2026, BLT hash n-gram embeddings, "Over-tokenized
transformer"). Difference: injected into the re-added x0 stream, funded by a value-embedding slot.
