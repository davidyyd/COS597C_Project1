# Output-side token metadata embedding
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: the head learns each token's output row independently; rare tokens get few updates. Sharing
statistical strength through token metadata (byte length x leading whitespace x first-character class) on
the OUTPUT side: W_eff[v] = W_out[v] + M[meta(v)] (M 192x512, zero-init) lets the model raise/lower whole
classes (e.g. "space-prefixed lowercase words after a word") at once. Confirm: < 0.95726.
Closest: factored/class-based softmax (Goodman 2001), our input-side token metadata embedding; difference:
an additive class embedding on the head rows (no hierarchical softmax).
