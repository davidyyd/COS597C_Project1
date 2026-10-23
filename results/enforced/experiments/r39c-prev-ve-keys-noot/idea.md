# Previous token's identity in keys, via the value-embedding table, normalized mixing
Parent: r38a-gated-proj
Type: exploit (output temperature removed for step time)

Prev-token keys helped early models (induction shortcut). Reuse the VE table rows of the previous token as a
key feature with normalized gated mixing: k <- norm(k) + tanh(W x) norm(VE[t-1]) (VE layers, signed gate 0 at
init), before RoPE and QK-norm. Confirm: < r38a.
