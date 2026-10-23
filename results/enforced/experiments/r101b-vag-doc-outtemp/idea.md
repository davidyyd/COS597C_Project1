# Cheap base (+RoPE/token) + document position in the output temperature (transfer test)
Parent: r101a-vag-rope-token
Type: explore (transfer test)
logits *= 1 + tanh(c f_doc), c init 0. Compare with r101a.
