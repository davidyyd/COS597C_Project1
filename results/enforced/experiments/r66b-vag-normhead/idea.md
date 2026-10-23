# Output embeddings normalized at use
Parent: r58b-va-gates-adam (0.94618)
Type: explore

The value-embedding fix (normalize sparse tables at use) suggests checking the other vocabulary table that is
weakly updated for rare tokens: the LM head (AdamW with weight decay every step). Logits = tau * x . norm(W_v)
(unit-RMS rows, learned global temperature tau, init 1/sqrt(512)); the 15-softcap stays.
Confirm: < r58b. Refute: worse (the row norms carry useful frequency information).
Closest work: cosine / normalized softmax classifiers, nGPT (normalized embeddings). Differs: only the output
embedding rows normalized at use in an otherwise standard model, motivated by the sparse-update argument.
