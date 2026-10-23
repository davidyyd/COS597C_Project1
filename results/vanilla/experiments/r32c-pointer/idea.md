# Pointer-sentinel copy head, computed exactly with two flex passes
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: copying is a big part of what this model learns (smeared keys/queries and contextual value
embeddings are among the largest wins), but a copy must still be routed through the vocabulary softmax.
A pointer head lets the model put probability directly on "the token that followed an earlier occurrence
of this context", with a learned sentinel deciding how much to trust the vocabulary softmax.

Model: final state x_t; pointer scores S_ts = (g ⊙ norm(Wq x_t)) · norm(Wk x_s) / 8 for s < t, sentinel
logit z_t = w · x_t; a = softmax over {s < t} ∪ {sentinel};
  p(y_t) = a_sent * p_vocab(y_t) + sum_{s<t} a_ts [token_{s+1} = y_t]   (a normalised distribution).
Computed exactly: lse_all and lse_match (keys whose successor equals y_t, via a score_mod on captured token
ids, no learnable captured tensors) from two 1-head, 64-dim flex passes; p = e^{z-L} p_vocab + e^{lse_match-L},
L = logaddexp(lse_all, z). Causal: only tokens <= t are used (verified on CPU: changing token t+1 changes no
loss at positions < t). The sentinel weight is init 0, pointer q/k random, gain 1.
Confirm: < 0.95726 (r25a). Refute: no gain (the softmax already copies well).
Closest work: pointer sentinel mixture models (Merity et al. 2016), copy mechanisms (Gu et al. 2016),
kNN-LM (retrieval mixture). Difference: exact in-context pointer over successor tokens with flex
attention LSEs, trained jointly in a modern small LM.
