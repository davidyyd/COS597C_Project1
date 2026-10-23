# RoPE base matched to each head's window (multi-scale heads)
Parent: r02d-multiscale-heads
Type: explore

Hypothesis: in the multi-scale sliding layers, a head with a 128-token window never sees distances beyond 128,
yet uses the same RoPE frequencies (base 10000, wavelengths up to ~60k) as the global head, so most of its
low-frequency channels are nearly constant over its window (wasted for position, only usable as content).
Scale each local head's base with its window: base_h = 10000 * window_h / 1024 (1250, 2500, 5000, 10000), so
each head's wavelengths cover the distances it can see. Full layers unchanged. No cost.
Confirm: < r02d (0.95520) by > 0.0005.
Closest work: per-head/learned RoPE frequencies (e.g. FIRE, learnable RoPE), YaRN (rescaling to context).
Differs: base tied to the head's own attention window.
