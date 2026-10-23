# d16m6 + attention gate computed from the full input (not x[:32])

Parent: r42a-d16m6-gate-fused. Explore: the head-wise output gate reads only 32 channels; give it the full d-dim input (Linear(d, n_head), zero init, Muon).
