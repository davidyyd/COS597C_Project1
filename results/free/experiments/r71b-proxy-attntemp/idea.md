# proxy backin5-mix + learned per-layer attention temperature (scalar on the 0.12 scale, init 1)

Parent: r67a. Per-head q gains were -0.0003 long ago; a per-layer scalar temperature lets depth-dependent sharpness differ (q scaled before attention).
