# Document masking in the local (sliding-window) layers only
Parent: r58b-va-gates-adam
Type: explore

Hypothesis: rows pack several documents; local (sliding-window) layers mostly need the current document, and keys
from the previous document right after a boundary are noise for them. Global layers (3, 7) may still use other
documents as a weak topic/style prior. So: mask cross-document keys only in the sliding layers.
Change: sliding layers: score(t, j) = -inf if j < doc_start(t) (flex score_mod with the doc starts, no params).
(Tested on the fast base: on the best model the score_mod costs ~1.3 ms and fails the time cap.)
Confirm: below r58b (0.94618) by > 0.0005.
Closest work: intra-document causal masking in all layers (Zhao et al. 2024; modded-nanogpt); here only local
layers are masked, global layers keep cross-document attention.
