# proxy: last 3 attention layers read the cached 2/3-depth activation (x_backout)

Parent: r59a-proxy-bf16resid (0.76505). From modded-nanogpt PR #241: feed the saved backout activation (residual after layer 2L/3) as the input of the last attention layers, so their inputs are not polluted by later MLP 'prediction' contributions; MLPs still read the full residual; backout subtraction unchanged.
