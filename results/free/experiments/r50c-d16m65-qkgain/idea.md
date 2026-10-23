# proxy: learnable per-channel QK-norm gains

Parent: r45d-d16m65-canon2-stash (0.76588). Explore: QK-norm currently has no gain; give q and k learnable per-head-dim gains (shape [head_dim], init 1, per layer), as in OLMo-2/Gemma QK-norm.
