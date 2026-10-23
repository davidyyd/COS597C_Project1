# d16m6 + elementwise (per-channel) attention output gate

Parent: r39a-d16-mlp6x-gskip (0.76913). Explore: Qwen's gated attention found an elementwise sigmoid gate (per channel) better than head-wise; gate 2*sigmoid(W x[:32]) of shape [B,T,d] applied to the attention output before c_proj (zero init = 1, W in Muon). Compare with head-wise gate r41d (0.76795).
