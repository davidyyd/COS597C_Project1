# Context-gated n-gram logit priors
Parent: r69a-r68a-split192-96. Type: exploit.
Hypothesis: the bigram/skip-bigram priors are context-blind (U[x_t] V^T is added at the same strength
everywhere); when the context already determines the next token the prior is noise. A per-token
confidence gate read from the final state lets the model turn each prior up or down.
Change: pg = 2*sigmoid(x[:32] @ G) (G: 32x2, zero-init => 1); feats = [x, pg0*U[x_t], pg1*U2[x_{t-1}]],
still one folded head matmul. Confirm: < 0.92658 (noise ~2e-4). Closest: mixture of n-gram and neural LMs
with a learned interpolation weight (e.g. neural cache / kNN-LM lambda); here the gate scales a low-rank
prior inside the logits, per prior order.
