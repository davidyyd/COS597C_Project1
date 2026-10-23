# Value embeddings also added to the keys (token identity for matching)
Parent: r18a-bf16-rope. Type: explore.

Hypothesis: as in r21a, but on the plain model: the keys of VE layers get a_h * VE(token), giving every
head an exact token-identity component to match on (copying / induction) at zero parameter cost.
Change: k <- k + a_h * ve, a_h per head init 0, in the 4 VE layers.
Confirm: < 0.95739 (r18a). Refute: no gain.
Closest work: value embeddings / value residuals (modded-nanogpt, ResFormer); token-identity key features
(smeared keys, Olsson et al.). Difference: reuse of the value-embedding tables as key embeddings.
