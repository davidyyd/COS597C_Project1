# Cheap lineage + multi-scale heads + x0 gate
Parent: r21c-cheap-gate-chain
Type: exploit

r21c (0.94791, 65.4 ms: doc temperature, scalar self-gate, full read chain) lacks two components of the best
model that are cheap: multi-scale head windows (also save time) and the x0 re-injection gate (+0.0016 when
removed from r14c). Confirm: < r21c by > 0.001.
