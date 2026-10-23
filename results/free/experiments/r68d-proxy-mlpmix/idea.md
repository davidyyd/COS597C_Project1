# proxy backin5 mix + MLP of the last 5 layers can also mix in the cached activation (init 0)

Parent: r67a-proxy-backin5-mix (0.76388). Symmetric extension: MLP input = x + n*(x_backout - x) with learnable n (init 0 = current behaviour).
