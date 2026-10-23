# Document-unigram cache prior folded into the head
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: tokens already used in a document are likely to recur (cache LM); a low-rank logit prior from
the running mean of a token table over the document captures this without attention.
Change: logits = [x, norm(docmean(C[x_s]))] @ [W_out, V_c]^T, C, V_c rank 64 (V_c zero-init).
Confirm: < 0.95726. Closest: unigram cache LMs (Kuhn & De Mori), neural cache (Grave et al.) which
uses attention over hidden states; here a low-rank bag-of-words logit prior, one extra head column block.
