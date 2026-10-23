# Heads compete per token (softmax over head outputs)
Parent: baseline
Type: explore

Hypothesis: with 4 heads, each token may need mostly one of them (local syntax vs. long-range retrieval);
independent heads all write. A per-token softmax over heads, scaled by H (= 1 each at init), forces a
budget: y_h <- H softmax_h(W x)_h * y_h, W [4, 512] init 0.
Confirm: < baseline by > 0.001. Refute: no gain (independent gates — self-gate r01b — are better).
Closest work: Mixture-of-Head attention (MoH, top-k routing of heads), gated attention (independent sigmoids).
Differs: dense softmax competition across heads, no routing.
