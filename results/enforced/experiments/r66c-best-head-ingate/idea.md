# Head input gate (feature selection for the readout)
Parent: r64a-ingate-instead (0.93169)
Type: explore

Input gates helped for the MLP and attention inputs. Apply the same per-channel gate (rank 32, AdamW) to the LM
head input: x <- x * 2 sigmoid(U V x) before the head (cheap: once). Confirm: < r64a.
