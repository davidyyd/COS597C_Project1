# Query-dependent null key (sink) computed exactly from the log-sum-exp
Parent: baseline. Type: explore.

Hypothesis: headwise output gates help a lot here (r05c -0.0054), i.e. heads need to abstain. A
principled abstain is a null key with value 0: softmax over keys + null gives y * sigmoid(lse - b),
so the head writes in proportion to how well the query matched anything (its log-sum-exp) relative to a
threshold. Making the threshold query-dependent b(x_q) combines match quality with intent.

Change: all layers flex_attention(return_lse=True); y_h <- y_h * sigmoid(lse_h - b_h(x)),
b_h(x) = c_h + w_h · x[:32] (c, w init 0).
Confirm: < baseline and ideally < r05c (0.9535).
Closest work: learned attention sinks (gpt-oss, per-head constant logit), softmax-plus-one, gated attention
(Qiu et al. 2025: gate independent of match quality). Difference: query-dependent sink via exact LSE.
