# Induction-candidate logit prior (attention-free copy cue)
Parent: r25a-bf16-resid. Type: explore.
Copying repeated sequences needs induction heads; here the candidate "token that followed the most recent
earlier occurrence of the current token" is computed exactly with a per-row stable sort (causal, cheap)
and fed to the logits through a rank-64 table (plus a 'none' row), folded into the head matmul.
Confirm: < 0.95726. Closest work: induction heads (Olsson et al.), n-gram caches, pointer networks.
Difference: an exact, attention-free last-occurrence lookup as a learned logit prior.
