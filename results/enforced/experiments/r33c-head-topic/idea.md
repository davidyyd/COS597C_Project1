# The LM head reads the row's topic vector
Parent: baseline
Type: explore

Hypothesis: a large part of next-token uncertainty is topical (which words are likely in this document). The
head reads only the last state; give it a direct, normalized causal sum of the token embeddings so far
(topic_t = norm(sum_{s<=t} x0_s)), x_head = norm(x_L) + beta topic_t (beta init 0): a cheap, context-dependent
unigram prior. The prefix sum is a lower-triangular matmul (exactly causal).
Confirm: < baseline by > 0.001.
Closest work: cache LMs / topic models (unigram cache), ctx-mean injection into the residual (r06c). Differs:
read only at the head.
