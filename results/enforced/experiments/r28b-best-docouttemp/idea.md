# Best model + doc-position output temperature
Parent: r26a-vegate-full
Type: exploit

logits *= 1 + tanh(a + b f_doc) (r20d: -0.002 on baseline). Confirm: < r26a by > 0.0005.
