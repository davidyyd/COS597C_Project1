# Output temperature also depends on document position
Parent: r73d-best-rope-per-window
Type: explore

Hypothesis: predictions early in a document (little context) should be less confident than late ones; the
content-based output temperature has to infer this from the residual. Document position already helps as a query
temperature (P5). Feed it to the output temperature too.
Change: logits *= 1 + tanh(w · norm(x) + c · f_doc), f_doc = log(1+t-doc_start)/log T, c scalar init 0.
Confirm: below r73d (0.93151) by > 0.0005.
Closest work: confidence calibration / temperature scaling; here a per-token, position-in-document learned scale.
