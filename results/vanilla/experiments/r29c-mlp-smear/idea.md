# Token shift for the MLP: the MLP also sees the previous position's normalized input
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: local context is the strongest lever found (contextual x0/VE, smeared keys, local early
attention), but the MLPs, which hold most parameters, only ever see one position. A per-channel smear of
the previous position's normalized input into the MLP input lets MLP features detect bigram patterns
directly at every layer.
Change: mlp(h_t + a ⊙ h_{t-1}), a per channel and layer, init 0.
Confirm: < 0.95726 (r25a). Closest work: RWKV channel-mix token shift, Canon layers (conv before MLPs).
Difference: a single learned lag-1 smear on the MLP input of a standard transformer.
