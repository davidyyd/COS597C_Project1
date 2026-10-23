# d22 b16 with fused ReLU2Proj + fused NormCanon (memory), LR 8e-3

Parent: r15b-d21-b16-fusedmlp-lr8e3. Exploit/scale: NormCanon autograd function saves only the residual input of each norm+canon (recomputes the normalized activation in backward), ~0.47GB/layer less, so d22 fits batch 16. Expected ~3310 train / ~3450 total.
