# r43d without Canon layers (ablation / time)

Parent: r43d-d16m65-gate. Canon (2 depthwise k=4 convs per layer) gave -0.003 early on but costs ~5-8% time; with bigram/trigram tables and gated skips its value may have shrunk. If close, the time buys width/depth.
