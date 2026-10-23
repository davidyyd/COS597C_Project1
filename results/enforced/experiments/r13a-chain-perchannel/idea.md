# Undiluted-read chain with per-channel weights
Parent: r12b-best-attn-reads-mlp
Type: exploit

r12b (attention reads norm(m_{l-1}), MLP reads norm(a_l), scalar weights) is best (0.94688). Per-channel
weights gave -0.0004 on r11a; apply to both reads. Confirm: < 0.9465.
