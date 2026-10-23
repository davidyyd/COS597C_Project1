# proxy: learned per-layer scale on the attention values (init 1)

Parent: r59a. modded-nanogpt's v_lambda grows to ~6-7; give each layer a learnable scalar on v (after VE mixing), init 1, AdamW scalar group.
