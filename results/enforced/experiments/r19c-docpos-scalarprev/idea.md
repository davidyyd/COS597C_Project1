# Scalar prev-token weight (cheaper)
Parent: r19b-scalargate-docpos
Type: exploit (time)

Per-channel 512-vector parameters on [B,T,512] activations cost ~1 ms each (their gradient reduction):
per-channel chain weights measured 72.7/79.8 ms. The prev-token key input uses a per-channel lambda; W_k can
select channels itself, so use one scalar per layer: k = W_k(u + lambda x0_{t-1}).
Confirm: same loss as r19b (within 0.0005) at lower step time.
