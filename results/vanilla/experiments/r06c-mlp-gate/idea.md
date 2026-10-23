# Per-token MLP output gate
Parent: baseline. Type: explore.

Hypothesis: per-token scalar control helps for the x0 mixing, the residual and attention heads; the MLP
branch may likewise benefit from a per-token write strength (e.g. tokens whose MLP features are
unreliable), which pre-norm otherwise only allows through the direction of its input.

Change: mlp(x) <- 2 sigmoid(w · x[:32]) mlp(x), w init 0.
Confirm: < baseline. Refute: no gain (the ReLU^2 MLP can scale its output itself).
Closest work: gated residual branches (ReZero, LayerScale are static); headwise gated attention.
