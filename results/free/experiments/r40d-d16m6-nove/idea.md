# d16 MLP6x proxy without value embeddings

Parent: r39a-d16-mlp6x-gskip. Ablation: value embeddings (8 tables + gates) cost time and memory; if their gain is small now (bigram/XSA/wide MLP), the budget could go to a deeper/wider model.
