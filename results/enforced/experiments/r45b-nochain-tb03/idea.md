# Token-conditioned MLP bias (layers 0-3) instead of the read chain
Parent: r44b-best-no-chain
Type: exploit

Token-conditioned MLP pre-activation biases from a normalized sparse table (r44c: -0.0021 on r43a, but ~7.7 ms in
all layers). Use them in layers 0-3 only, with the time freed by dropping the read chain.
Confirm: < r44b (0.93926); compare with the chain (r41b, 0.9369).
