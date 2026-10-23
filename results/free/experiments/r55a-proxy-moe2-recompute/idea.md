# proxy with 2-expert MoE MLP (hidden 10752, cf 1.1), relu^2 recomputed in backward

Parent: r45d-d16m65-canon2-stash. Retry of r54a (OOM): the expert MLP now uses a batched ReLU2-proj autograd function that saves only the pre-activation (like the dense MLP's ReLU2Proj). Compare with dense 6x control r54b (0.76734, 3208s).
