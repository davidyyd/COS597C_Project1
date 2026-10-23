# MLP reads attention + multi-scale heads (two base wins)
Parent: r10c-mlp-reads-attn
Type: exploit

Combine the two largest single changes on the baseline: MLP reads the fresh attention output (r10c,
-0.0052) and multi-scale head windows (r02d, -0.0037; also saves time, which later additions need).
Confirm: < r10c (0.95379) by > 0.001.
