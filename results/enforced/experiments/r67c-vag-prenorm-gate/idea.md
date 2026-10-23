# Per-channel gate before the MLP's normalization
Parent: r58b-va-gates-adam
Type: explore

Gate the residual channels before the MLP's RMSNorm (x * 2 sigmoid(U V norm(x)), then norm): the gate decides which
channels dominate the normalization (relative scale), unlike the input gate after the norm (r63c). Compare.
