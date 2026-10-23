# Low-rank elementwise gate on the MLP input
Parent: r25a-bf16-resid. Type: explore.
The low-rank gate on the MLP hidden units helped (r39c, -0.0034) but is expensive (2048-wide). Gating the
512-wide MLP input per channel (feature selection per token, rank-32 read) is 4x cheaper; tests whether
most of the benefit is token-dependent feature selection.
Change: mlp(x ⊙ 2 sigmoid(G x[:32])), G ∈ R^{512x32} init 0. Confirm: < 0.95726 (r25a).
