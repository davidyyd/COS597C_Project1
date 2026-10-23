# Output temperature from the position within the document
Parent: baseline
Type: explore

Hypothesis: r19d showed attention sharpness should depend on how far into its document a token is. The same
should hold for the output distribution: right after a document starts, predictions should be flatter;
deep in a document, sharper. logits *= 1 + tanh(a + b f_t), f_t = log(1 + t - doc_start)/log(T), a, b init 0
(two scalars). Confirm: < baseline by > 0.001.
Closest work: temperature scaling (global, post hoc); per-token out-temperature (r04c, from content).
Differs: two scalars driven only by the document position.
