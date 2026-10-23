# d16m6 proxy: input-dependent gates on the x0 re-injection

Parent: r39a-d16-mlp6x-gskip (0.76913). Explore (MUDD-lite, after gated skips gave -0.0013): x0 (embedding incl. bigram) re-added at each layer with a per-layer dynamic gate 2*sigmoid(w_i . x[:32]) (zero init) times the learned scalar.
