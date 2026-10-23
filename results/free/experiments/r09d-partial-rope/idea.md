# Half-truncated RoPE (modded-nanogpt): 32 freqs from 1 to 1/1024, other half of dims unrotated

Parent: r07d-canon. Explore (positional encoding). modded-nanogpt's rotary: frequencies (1/1024)^linspace(0,1,head_dim/4) for half the pairs and zero frequency (no rotation, pure content channels) for the other half.
