# Canon layers: causal depthwise conv (k=4) before attention and MLP

Parent: r06d-attnscale. Explore (architecture). Allen-Zhu 'Physics of LMs 4.1' Canon layers: a residual short causal depthwise convolution over the token axis (kernel 4) on the normalized input of each attention and MLP sublayer, cheap local token mixing. Conv weights zero-init (identity at start), AdamW.
