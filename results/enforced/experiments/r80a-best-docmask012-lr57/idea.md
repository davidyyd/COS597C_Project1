# Best + document masking in the early local layers (0-2), low-rank head gate in 5, 7
Parent: r77b-token-head-mlp
Type: exploit
Hypothesis: on the fast base, doc masking in local layers 0-2 gave -0.0028 while masking 4-6 HURT (+0.0013): early
layers should build features from the current document only; late layers use other documents. On the best model
(r79b) masking all local layers was net ~-0.0002. Masking only 0-2 should recover the full gain.
Time: low-rank head gate only in layers 5, 7.
Confirm: below r77b (0.93066) by > 0.0005.
