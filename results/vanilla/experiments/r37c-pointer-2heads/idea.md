# Two-head pointer: one sentinel, jointly normalised over both heads' keys
Parent: r36d-pointer-chunked. Type: explore.

Hypothesis: copying needs different match criteria (recent local repeats vs. distant named entities /
long repeated spans). Two pointer heads with separate q/k projections, normalised jointly with the
sentinel (softmax over {sentinel} ∪ {(s, h)}), give a richer copy distribution at ~2x pointer cost.
Confirm: < 0.953060 (r36d).
