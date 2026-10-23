# Attention temperature from the position within the document
Parent: baseline
Type: explore

Hypothesis: the doc-position feature helped (r18d, -0.0021). Its natural use is in attention: right after a
document starts there is little in-document context, so heads should attend differently (broader, or
toward BOS) than deep into a document. Per-head temperature from the doc position only:
q_h <- norm(q_h) * (1 + tanh(c_h f_t)), f_t = log(1 + t - doc_start)/log(T), c_h init 0 (4 scalars/layer).
Confirm: < baseline by > 0.001; compare with r18d (input feature).
Closest work: log-n attention scaling (SSMax, YaRN's temperature: position-dependent, fixed); here learned per
head and measured from the document start.
