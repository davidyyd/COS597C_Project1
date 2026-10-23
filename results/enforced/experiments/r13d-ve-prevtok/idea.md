# Bigram values: the first value-embedding layer indexes the previous token
Parent: baseline
Type: explore

Hypothesis: value embeddings give the attended position's token identity, which the residual already has
(x0 re-injection). The identity of the token BEFORE the attended one is new information for values (what
preceded the matched context), and prev-token keys helped (r01a). Switch layer 1's value-embedding table to
be indexed by token t-1 (no new parameters); layers 3, 5, 7 unchanged.
Confirm: < baseline by > 0.001. Refute: worse (current-token identity in values matters more).
Closest work: value embeddings (modded-nanogpt), smear / bigram hash embeddings (input side). Differs:
previous-token identity injected into the values of one layer.
