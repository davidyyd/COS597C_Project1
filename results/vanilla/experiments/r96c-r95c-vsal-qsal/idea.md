# r96a + per-query temperature (5th gate block)
Parent: r95c-r94b-gatesall-ve3 (via r96a). Type: exploit. Completes the per-token attention sharpness controls:
q_t *= 2 sigmoid(w_h . x_t[:32]) after QK-norm (with the per-key salience and the position temperature).
