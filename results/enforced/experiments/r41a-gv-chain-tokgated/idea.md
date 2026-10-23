# Token-gated undiluted reads (cheap lineage)
Parent: r40b-gv-chain
Type: exploit

The four undiluted reads use one scalar weight each. Following the gated-mixing principle that made the value
path work, modulate each read per token: weight_j * (1 + tanh(w_j . norm(x))) (w init 0), one 512x4 map per
block. Confirm: < r40b (0.94330) by > 0.0005.
