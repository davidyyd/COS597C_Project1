# Per-key value salience
Parent: r25a-bf16-resid. Type: explore (companion to r92d key salience).
v_s *= 2 sigmoid(w_h . x_s[:32]) (zero-init gives 1). Unlike key salience (which changes the softmax competition),
this rescales a key's contribution after normalization, letting tokens be "read" weakly even when attended.
Unlike the output gate (query side), it is per key. Confirm: < 0.95726; compare to key salience 0.955189.
