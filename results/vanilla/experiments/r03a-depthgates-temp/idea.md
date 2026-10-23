# Depth gates (residual + x0) + per-head temperature, cheaper gate read
Parent: r02a-x0gate-temp (adds r02b's residual gate). Type: exploit.

Hypothesis: r02b's per-token residual gate gave -0.0021 over r01c but ran 265 s (over the time cap);
fusing both gates into one 32x2 matmul read from norm(x[:32]) (RMS of the slice only, instead of a full
512-wide norm) should keep the gain within the time cap, and add to the temperature of r02a.

Change: g = 2 sigmoid(norm(x[:32]) W_i), W_i in R^{32x2} init 0; x <- r_i g_1 x + l_i g_0 x0.

Confirm: < 0.9519 (r02a) with training_seconds < 262.
Closest work: highway/LSTM depth gating; hyper-connections. Difference: two per-token scalars only.
