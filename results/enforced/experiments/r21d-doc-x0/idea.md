# Token-identity re-injection modulated by the position within the document
Parent: baseline
Type: explore

Hypothesis: near the start of a document a token's identity is most of what is known; deep into a
document, context dominates. Make the x0 re-injection weight depend on the doc position:
x <- r_l x + a_l (1 + tanh(c_l f_t)) x0, f_t = log(1 + t - doc_start)/log(T), c_l init 0 (8 scalars).
Confirm: < baseline by > 0.001.
Closest work: x0 lambdas (static per layer), the content-based x0 gate (r04d). Differs: driven by the
document position only.
