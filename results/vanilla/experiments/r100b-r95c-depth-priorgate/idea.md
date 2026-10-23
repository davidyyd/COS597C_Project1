# Prior gates conditioned on n-gram repetition depth
Parent: r95c-r94b-gatesall-ve3. Type: exploit.
When the current n-gram is a repeat (copy mode), the attention can predict the continuation and the static
n-gram logit priors are less useful (or harmful); let the repetition depth (0..4) shift the 12 prior gates:
pg = 2 sigmoid(x[:32] G + D[depth]) (D zero-init). Confirm: < 0.921402.
