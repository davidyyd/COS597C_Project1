# Best - low-rank head gate + local document masking (score_mod)
Parent: r77b-token-head-mlp
Type: exploit
Hypothesis: the low-rank head self-gate is worth ~0.0009 and costs ~1.6 ms; local document masking was worth -0.0019
on the fast base and costs ~1.3 ms as a score_mod. Trading one for the other should be a net gain.
Change: lowrank gate removed; sliding layers: score = -inf if j < doc_start(t).
Confirm: below r77b (0.93066).
