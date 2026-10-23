# d16m6 proxy: head-wise attention output gate

Parent: r39a-d16-mlp6x-gskip. Explore: retest a head-wise sigmoid gate on the attention output (Qwen gated attention), from x[:32] like the VE gate, zero init (gate 1); dynamic gates (skips) helped recently. Was neutral in the tiny baseline long ago.
