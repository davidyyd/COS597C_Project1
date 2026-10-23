# Document-length-aware attention temperature
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: the number of keys a query competes over grows with its position in the document, so the
softmax gets flatter as the document grows; a per-head learned sharpening q *= 1 + a_h log(1 + n_doc)/log(T)
(n_doc = tokens since the last BOS, a_h zero-init) keeps attention focused. Confirm: < 0.95726.
Closest: Scalable-Softmax / log-n attention scaling (fixed s log n over absolute position); difference:
per-head learned slope and document-relative length in packed rows.
