# Output embeddings coupled to input embeddings by a learned low-rank map
Parent: baseline
Type: explore

Hypothesis: input and output embeddings encode related token semantics, but the untied head learns its
rows from scratch (rare tokens get few updates); full tying hurts at this scale. Couple them softly:
W_out = W_head + E_in A B with A [512,128], B [128,512] (B init 0), so the head can borrow structure the
input embedding learned, through a learned rank-128 map. Cost per step, not per token.
Confirm: < baseline by > 0.001.
Closest work: weight tying (Press & Wolf), untied/retied schedules (modded-nanogpt). Differs: a learned
low-rank linear coupling added to an untied head.
