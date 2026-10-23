# Cheap base (+RoPE/token) + time-shifted key read in the non-isolated layers (transfer test)
Parent: r101a-vag-rope-token
Type: explore (transfer test)
Keys read u_t + b u_{t-1} (b per layer, init 0) in layers 3-7 (the isolated layers do not, as in the best model).
Compare with r101a.
