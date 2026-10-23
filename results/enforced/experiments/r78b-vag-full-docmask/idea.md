# Control: document masking in all layers (fast base)
Parent: r58b-va-gates-adam
Type: explore (control for r78d)
Known method (intra-document causal masking) run as the reference for r78d, which masks only the local layers.
If r78d >= r78b in quality, cross-document attention in the global layers is useful; if r78b is better, every
layer should stay within its document.
Change: all layers use flex_attention with score(t, j) = -inf if j < doc_start(t).
