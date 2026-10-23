# proxy backin5: token-dependent mix gate sigmoid(w.x[:32]) instead of a scalar

Parent: r67a-proxy-backin5-mix (0.76388). Dynamic gates helped elsewhere (skips, attn gate); make the cached/current mix token-dependent: m = sigmoid(w . x[:32] - 4) (zero-init w, starts ~0.02).
