# r69a without U-net, plus a mid-network gated hashed bigram memory
Parent: r69a-r68a-split192-96 (via r76a). Type: exploit (combine).
r76a showed U-net is worth ~0 in r69a; r76d showed the gated mid-network hashed bigram memory gives
-0.001 on the control. Swap: drop U-net, inject mem_proj(T[hash(x_{t-1},x_t)]) at layer 4 with a gate
2*sigmoid(h[:32]·w) read from the previous block's normalized MLP input (no extra norm).
Confirm: < 0.92658 within the 261.66 s cap. Refute: no gain (priors already cover the bigram).
Closest: Engram (conditional n-gram memory), hashed n-gram embeddings.
