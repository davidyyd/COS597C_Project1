# Low-rank self-gate in all layers, its matrices trained as gates (AdamW)
Parent: r58a-best-gates-adam (0.93404)
Type: exploit

The per-dimension low-rank self-gate was restricted to some layers because it cost ~1 ms per 2 layers; most of
that cost was Muon on its many tiny [32,128] matrices. Its matrices are gate parameters, so label them "other"
(AdamW) and use it in every layer. Confirm: < r58a.
