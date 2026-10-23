# Per-dimension self-gate (low-rank) on attention heads
Parent: r05b-best-x0gate
Type: exploit

The head self-gate 2 sigmoid(a_h . y_h) is one scalar per head. Extend to a per-dimension gate computed from
the retrieved vector through a rank-32 map: y_h <- y_h * 2 sigmoid(a_h . y_h + U_h V_h y_h), U_h init 0
(V_h [32,128] uniform), so the head can keep some value components and drop others depending on content
(a GLU on the retrieved value). Confirm: < r05b by > 0.0005.
Closest work: GLU, gated attention (elementwise gate from x). Here from the attention output itself.
