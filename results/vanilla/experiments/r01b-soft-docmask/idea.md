# Soft document mask: learned per-head cross-document logit bias
Parent: baseline. Type: explore.

Hypothesis: rows pack several documents; the baseline attends freely across document boundaries, which
dilutes attention with unrelated context. Hard document masking is known to help; but some heads may
benefit from cross-document context (e.g. generic statistics, a sink). Let each head learn how much to
suppress cross-document keys.

Change: logit(q,k) += b_h * [doc(q) != doc(k)], b_h learned per head and layer (init 0), doc = cumsum(BOS).
All layers now use flex_attention (causal block mask for the long layers).

Confirm: val_bpb clearly below baseline; inspect whether learned b_h become strongly negative
(-> hard masking) or stay moderate in some heads.

Closest work: document masking (modded-nanogpt, Llama 3 intra-document masking). Difference: soft, learned
per head rather than a hard mask.
