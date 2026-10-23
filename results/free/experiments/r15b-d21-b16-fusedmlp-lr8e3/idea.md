# d21 at device batch 16 with fused ReLU^2-proj (recompute activation in backward), LR 8e-3

Parent: r14b-d21-win256-lr6e3. Explore/exploit (efficiency): custom autograd ReLU2Proj saves only the pre-activation (not relu^2, 470MB/layer at b16) so d21 may fit batch 16 (~5% faster than b8). Risk: OOM.
