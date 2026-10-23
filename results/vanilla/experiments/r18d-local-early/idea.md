# Local early layers: layers 0 and 1 use 128-token attention windows
Parent: baseline. Type: explore.

Hypothesis: the first layers see token identities with little context and mostly build local features;
long windows there add noise and cost. r17a found a 128-window layer 1 better than a 1024 window. Tested
here on the baseline (layers 0-1 window 128, rest unchanged), as an alternative to the conv layer 0
(r08d: -0.0035).
Confirm: < baseline. Refute: worse (early long-range attention matters).
Closest work: local-to-global layer schedules (Longformer, Gemma-style local/global interleaving).
Difference: local only at the bottom with a very short window.
