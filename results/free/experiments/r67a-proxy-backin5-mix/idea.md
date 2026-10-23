# proxy backin5 + learnable per-layer mix between cached and current residual (init 0)

Parent: r65c-proxy-backin5 (0.76423). Generalize: attention input = x_backout + m*(x - x_backout) with m learnable per block (init 0 = pure backout input).
