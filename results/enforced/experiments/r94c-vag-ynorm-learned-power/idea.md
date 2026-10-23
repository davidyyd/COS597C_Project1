# Learned partial output normalization per head (y / rms(y)^a_h)
Parent: r91c-vag-headgate
Type: explore
Hypothesis: output normalization helps the cheap base but hurts the best model; each head may want a different amount
of the agreement magnitude. y_h <- y_h / rms(y_h)^a_h with a_h learned (init 0 = unchanged; 1 = unit direction).
Confirm: below r91c (0.93837) and close to r92b (0.93737, full normalization + scale).
