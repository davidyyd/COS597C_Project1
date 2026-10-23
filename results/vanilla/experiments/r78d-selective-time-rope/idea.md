# Selective-time RoPE: token-dependent position clock per head
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: RoPE measures distance in tokens, but token count is a poor clock (a long word split into
pieces vs a short word; whitespace/punctuation). Let each head advance its own clock by a learned
token-dependent step: p_{t,h} = sum_{s<=t} dt_h(x_s), dt_h = 2 sigmoid(w_h . x0_s) (zero-init => p = t,
exact RoPE at init). Rotation angles are p_{t,h} * inv_freq; relative phase between q_t and k_s is
the sum of dt over (s, t]. Causal (cumsum). Confirm: < 0.95726. Closest: CoPE (contextual position via
query-key gates over keys, O(T^2)); Mamba's selective dt. Difference: an O(T) cumulative clock fed into
standard RoPE, no change to attention.
