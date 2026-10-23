# Pointer-sentinel copy head (dense implementation)
Parent: r25a-bf16-resid. Type: explore.

Same model as r32c/r33c (which crashed in Inductor's flex lowering), computed with plain dense ops:
S = (g ⊙ norm(Wq x_t)) · norm(Wk x_s) / 8 over (B, T, T); lse_all / lse_match are masked logsumexps with
finite -1e4 masking (no NaN gradients, masked positions get no gradient);
  p(y_t) = e^{z-L} p_vocab(y_t) + e^{lse_match - L},  L = logaddexp(lse_all, z),
a normalised mixture of the vocabulary softmax and a copy distribution over the successors of earlier
positions (pointer sentinel mixture). Strictly causal (CPU check: changing token t+1 changes no loss at < t).
Confirm: < 0.95726 (r25a). Refute: no gain.
Closest work: pointer sentinel mixture models (Merity et al. 2016), copy mechanisms; here as an exact
in-context successor pointer in a modern small LM.
