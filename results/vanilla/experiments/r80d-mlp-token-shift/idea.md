# Selective token shift on the MLP input
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: the MLP sees one position; letting it read a per-channel, per-token gated mix of the previous
position's normalized state gives it bigram-level features at no attention cost (selective q/k smears
were among our strongest wins; the MLP has no such path).
Change: h' = h + s_t mu (h_{t-1} - h_t), s_t = 2 sigmoid(w . h_t[:32]), mu (512) zero-init.
Confirm: < 0.95726. Closest: RWKV channel-mix token shift (static per-channel mix); difference: a
per-token selective gate (like our selective smear) on a pre-norm transformer's MLP.
