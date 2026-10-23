# r88c-r87b-dist-nometa + position-aware attention temperature (cheap static form)
Parent: r88c-r87b-dist-nometa (0.923063 @ 261.3). Type: exploit.
r91d showed that the temperature gain on the control comes from position, not document length (row 0.954847 =
doc 0.954848). In the stack the doc-length version cost ~1.6 s (per-batch (B,T) log-length tensors per layer);
the row version needs only a static (T,) table: q *= 1 + a_{l,h} log(1+t)/log(T), a zero-init per head.
Confirm: lower val_bpb than the parent within the time cap. Closest: Scalable-Softmax (s log n, fixed);
ours has a learned per-layer-per-head slope on top of QK-norm and query gains.
