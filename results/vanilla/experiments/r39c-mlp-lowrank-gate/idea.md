# Low-rank elementwise gate on the MLP hidden units
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: gates with a tiny 32-channel read have been consistently strong (x0/residual gates, head and
channel output gates). The MLP is ungated (ReLU^2 only); a cheap per-hidden-unit gate computed from 32 input
channels (rank-32, unlike a full GLU) lets the token select which features are active.
Change: mlp(x) = W2 (relu(W1 x)^2 ⊙ 2 sigmoid(G x[:32])), G ∈ R^{2048x32} init 0.
Confirm: < 0.95726 (r25a). Closest work: GLU/SwiGLU (full-rank gate, more FLOPs). Difference: rank-32 gate.
