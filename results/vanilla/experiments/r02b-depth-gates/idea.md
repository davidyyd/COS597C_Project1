# Per-token depth gates: gate both the residual and the x0 re-injection
Parent: r01c-x0-token-gate. Type: exploit.

Hypothesis: r01c showed per-token control of the x0 re-injection helps. The residual carry is the other
half of the same mixing; letting each token also rescale its carried residual (relative to x0 and the
new block output) gives a learned per-token "forget" across depth.

Change: x <- r_i * gr_i(x) * x + l_i * g_i(x) * x0, gr_i, g_i = 2 sigmoid(w · norm(x)[:32]) (init 0 => 1).

Confirm: below r01c. Refute: no gain (residual scale is irrelevant under pre-norm).
Closest work: highway networks / LSTM-style depth gating, hyper-connections. Difference: scalar
per-token gates on the two inputs of the x0-mixing, read from 32 channels.
