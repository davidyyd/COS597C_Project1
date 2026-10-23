# proxy (bf16 resid): value residual from layer 1 with dynamic per-head gate

Parent: r59a-proxy-bf16resid. As r63a but the mix weight is sigmoid(W x[:32]) per head (zero init = 0.5), like the VE gate.
