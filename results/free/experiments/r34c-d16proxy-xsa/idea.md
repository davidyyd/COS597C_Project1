# d16 proxy + learnable XSA (bf16)

Parent: r34b-d16proxy. Explore: learnable exclusive self-attention (modded-nanogpt #82), bf16 formulation y <- y - tanh(a_h) (<y,v>/<v,v>) v; previous attempts OOMed on the memory-heavy old proxy.
