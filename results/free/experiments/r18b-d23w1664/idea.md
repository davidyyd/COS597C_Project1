# Deeper/narrower at ~same FLOPs: d23 x 1664 (13 heads)

Parent: r17a-d21-b16-SL-wu2. Exploit (shape): depth has kept paying; 23 layers x 1664 has 5.6% fewer matmul FLOPs than 21 x 1792 (more per-layer elementwise/attention overhead).
