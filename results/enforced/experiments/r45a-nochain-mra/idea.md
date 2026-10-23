# One-read chain: only the MLP reads the fresh attention output
Parent: r44b-best-no-chain
Type: exploit

The full 4-read chain is worth ~0.0023 on the best model but costs ~5 ms (r44b). The first read alone (MLP reads
norm(a_l)) was the largest single gain originally (r10c). Test whether it recovers most of the chain's value at
a fraction of the time. Confirm: < r44b (0.93926) by > 0.0015.
