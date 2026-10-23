# Best model without the x0 gate
Parent: r14c-chain3-attn-reads-attn
Type: exploit (simplification / time)

The per-token x0 re-injection gate may be redundant now that sublayers read fresh outputs directly; removing
it saves a norm per layer. Confirm "redundant": within +0.0005 of r14c (0.94455).
