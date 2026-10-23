# Best model + per-token output temperature
Parent: r03b-qtanh-prevtok
Type: exploit

Add the output temperature logits *= 1 + tanh(w . norm(x)) (r04c: -0.0025 on baseline) to the best model.
Both q-temperature and out-temperature restore a magnitude channel erased by a gainless norm, at
different places, so they should add. Confirm: < 0.9520.
