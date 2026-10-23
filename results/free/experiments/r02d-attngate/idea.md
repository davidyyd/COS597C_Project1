# Gated attention (head-wise sigmoid output gate)

Parent: r01c-grouplr. Explore (architecture). Qwen 'Gated Attention' (2025): an input-dependent sigmoid gate on the SDPA output removes attention sinks and lowers loss. Head-wise gate 2*sigmoid(W x) with W (n_head x n_embd), zero init (gate=1 at init), trained by Muon.
