# Multi-scale RoPE: a different fixed base per head
Parent: baseline. Type: explore.

Hypothesis: all heads share one RoPE spectrum, so every head has the same positional resolution and the
same range of position-insensitive (low-frequency) channels. Heads specialise (local vs. long-range), and
the learned per-head RoPE scale (r06d, -0.0033, but slow to train) suggests benefit. Fixed per-head bases
spread geometrically (1e3, 4.6e3, 2.2e4, 1e5) give each head a different resolution at zero cost.

Change: inv_freq_{h,j} = base_h^{-2j/d}, base_h = 1e4 * 10^{(h-1.5)/1.5}.
Confirm: < baseline. Refute: no gain / worse.
Closest work: per-head/learnable RoPE frequencies (various 2024-25 variants), partial RoPE, NoPE heads.
Difference: fixed geometric spread of bases across heads, no parameters.
