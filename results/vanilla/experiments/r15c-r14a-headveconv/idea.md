# r14a with per-head (not per-channel) contextual value-embedding conv weights
Parent: r14a-r13b-ctxve. Type: exploit.

Per-channel conv weights need a column reduction over B*T for their gradient (Inductor fused it into a
pathological 8 ms kernel in r13d). With per-head lag weights (4 lags x 4 heads per table) the gradient
is a reduction to 16 numbers, and the hypothesis is that lag weighting does not need channel detail.
Confirm: ≈ 0.939 and faster than r14a.
