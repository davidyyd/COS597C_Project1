# Multi-timescale causal context channels
Parent: r01b-attn-selfgate (the baseline version measured 70.6 ms)
Type: explore

Hypothesis: the causal running mean of token embeddings (r06c, -0.0014) helped as a free topic signal, but a
single uniform average mixes everything since the row start. Use four timescales: split x0's channels into 4
groups and average group g causally with an exponential kernel of half-life 8 / 64 / 512 / 4096 tokens (phrase,
sentence, paragraph, document). c = norm(EMA(x0)), re-injected before every block with b_l (init 0). One
[T,T] lower-triangular matmul per group (fixed kernels, +1.4% FLOPs).
Confirm: < r01b (0.95653) by > 0.001.
Closest work: multi-scale EMA (MEGA), long convolutions, CBOW context. Differs: fixed multi-timescale averages
of the embeddings re-injected like x0.
