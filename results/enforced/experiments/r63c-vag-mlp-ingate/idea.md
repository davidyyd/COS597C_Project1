# Per-channel MLP input gate from the input itself
Parent: r58b-va-gates-adam (0.94618)
Type: explore

The write gate (output side, from the input) helped (r60b, -0.0016). Test the input-side analog: the MLP input is
gated per channel by a rank-32 map of itself before W1 (u <- u * 2 sigmoid(U V u)), i.e. multiplicative feature
selection. Compare with r60b.
