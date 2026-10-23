# r09c with layers 0 and 1 both as value-convolution layers
Parent: r09c-conv0-all. Type: exploit.

Hypothesis: if layer-0 attention was wasteful (r08d), layer 1 (which sees only one layer of processing)
may also mostly do local mixing; replacing its attention too (keeping its value embedding) saves time
and may not hurt. Tests how far the "early layers are local" principle goes.
Change: layer 1 also y = c_proj(dwconv4(v + gate*ve)).
Confirm: ≈ or < 0.9445 with less time. Refute: clearly worse => layer 1 needs attention.
