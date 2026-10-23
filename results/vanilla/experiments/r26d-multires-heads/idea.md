# Multi-resolution heads: half the heads of each sliding-window layer are local (128)
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: local early layers helped (r18d), but making a whole layer local (layer 2, r19b) hurt: the
model needs both local and mid-range context at most depths. Splitting resolution across heads instead of
across layers gives every sliding-window layer 2 local heads (window 128) and 2 long heads (1024): local
heads specialise in syntax/n-gram structure, the others in mid-range context, and the attention cost drops.
Change: per-head window in the flex block mask of the S layers: w_h = 128 for h < 2, else 1024.
Confirm: < 0.95726 (r25a). Refute: no gain / worse.
Closest work: Longformer/BigBird (local+global patterns per head), multi-scale attention heads
(e.g. "MSA" variants). Difference: a fixed per-head split inside each sliding-window layer of a small LM.
