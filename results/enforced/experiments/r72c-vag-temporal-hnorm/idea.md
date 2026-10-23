# Temporal MLP input gate + MLP hidden normalization (fast base)
Parent: r71d-vag-temporal-ingate
Type: exploit

Hypothesis: the two round-71 winners (temporal input gate -0.0021, hidden normalization -0.0020) act on different
ends of the MLP (input selection vs output magnitude) and should stack.
Change: r71d with m = W_proj norm(relu(W_fc u)^2).
Confirm: below r71d (0.94226) by > 0.001.
