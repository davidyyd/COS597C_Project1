# Best model: swap the x0 gate for "MLP reads the fresh attention output"
Parent: r10b-qfeat-gate
Type: exploit

r10b + MLP-reads-attention exceeded the step cap (69.4 ms). Both the x0 gate and MLP-reads-attention give
sublayers undiluted access to a signal; drop the x0 gate (~1 ms, worth ~-0.0007) and add
u = norm(x + a) + beta_l norm(a) (-0.0052 on baseline). Confirm: < r10b (0.94946).
