# Attention input gate computed from the token embedding
Parent: r64d-vag-attn-ingate (0.94343)
Type: explore

Is the attention input gate's benefit about context-dependent feature selection (gate from the input itself, r64d)
or about token-dependent selection? Compute the gate from x0 (token identity) instead: u <- u * 2 sigmoid(U V x0).
Compare with r64d.
