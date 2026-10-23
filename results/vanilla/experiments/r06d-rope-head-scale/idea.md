# Per-head learned RoPE frequency scale
Parent: baseline. Type: explore.

Hypothesis: all heads share one RoPE spectrum (base 10000), but heads specialise: local heads (previous
token, n-gram) want high positional resolution, long-range/semantic heads want position-insensitive
matching. A single learned scale per head on all frequencies lets each head choose its resolution.

Change: angle_{t,h,j} = t * inv_freq_j * exp(s_h), s_h learned (init 0).
Confirm: < baseline, and learned s_h spread. Refute: no gain.
Closest work: learnable RoPE frequencies (various), NTK/YaRN scaling (fixed, for extension), partial
RoPE. Difference: one scalar per head learned during training.
