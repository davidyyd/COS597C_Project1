# Character n-gram embeddings of the tokens (FastText-style subword sharing)
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: BPE token embeddings are independent parameters, so rare tokens get poor embeddings even when
they share spelling with frequent ones ("Tokyo" / " Tokyo" / "Tok"). Adding the mean of hashed byte 3-gram
embeddings of each token's string (4096 shared buckets, 2.1M params) lets embeddings share statistical
strength across similarly spelled tokens. Computed per vocabulary entry with an embedding_bag (cheap), then
looked up by token id; the token-metadata result (r48d, -0.0016) suggests spelling features help.
Change: x0 = norm(wte[t] + mean_{g in ngrams(t)} N[hash(g)]), N init 0.
Confirm: < 0.95726 (r25a). Closest work: FastText subword embeddings, char-aware LMs (Kim et al. 2016).
Difference: hashed n-gram embeddings layered on top of a BPE embedding in a modern small LM.
