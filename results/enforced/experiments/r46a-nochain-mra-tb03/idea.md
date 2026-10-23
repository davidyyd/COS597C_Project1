# Cheaper replacement of the read chain: MLP-reads-attention + token MLP bias (layers 0-3)
Parent: r45b-nochain-tb03 (= best model without the read chain + token MLP bias in layers 0-3)
Type: exploit

Each of the two cheap pieces gave ~-0.001 over the chain-free model (r45a, r45b); together (~3.3 ms) they may
match or beat the full 4-read chain (~5 ms). Confirm: < r41b (0.9369 mean of two runs).
