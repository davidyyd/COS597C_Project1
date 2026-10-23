# proxy backin5-mix + attention lane: last layers' attention reads x_backout + accumulated attention outputs (no MLP outputs)

Parent: r67a-proxy-backin5-mix (0.76388). Toward partitioned hyperconnections (modded-nanogpt #73): instead of a frozen cache, the attention input lane for the last 5 layers is x_backout plus the attention outputs of those layers (MLP 'prediction' outputs excluded), mixed with the full residual by the learnable m (init 0).
