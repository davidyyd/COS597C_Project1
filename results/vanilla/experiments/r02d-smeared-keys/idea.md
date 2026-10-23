# Learned per-channel smeared keys
Parent: baseline. Type: explore.

Hypothesis: induction/copy heads need keys that encode the previous token; in the baseline this requires
a previous-token head in an earlier layer. Mixing the previous position's key into the current key
(per channel, learned) makes the "find where my previous token occurred" pattern one-layer.

Change: k_t <- k_t + a ⊙ k_{t-1} (a: per head-channel vector, init 0), before RoPE/QK-norm.

Confirm: val_bpb below baseline. Refute: no gain.
Closest work: smeared keys (Olsson et al. 2022), RWKV token shift, modded-nanogpt embedding smear.
Difference: per-channel learned smear on keys only, in every layer.
