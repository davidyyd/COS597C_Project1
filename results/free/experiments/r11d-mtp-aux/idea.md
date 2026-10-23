# Multi-token prediction auxiliary loss (t+2 head, weight 0.3)

Parent: r10d-polar-express. Explore (objective). Gloeckle et al. 2024 / DeepSeek-V3: an auxiliary next-next-token loss improves data efficiency. Here a light head: x2 = norm(x + W x) (W zero-init) -> shared lm_head -> soft-cap -> CE vs targets shifted by one, added with weight 0.3 to the training loss only (eval unchanged).
