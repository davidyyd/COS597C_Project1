# Hard-wired induction prior in the global layers via an SDPA additive mask
Parent: r25a-bf16-resid. Type: explore (the flex score_mod versions failed to compile).
Fixed per-head bonus beta_h * [token_{s-1} == token_t] (beta = 0, 1, 2, 4) added to the attention scores of
the two global layers through a dense (B,H,T,T) additive mask (expensive, but tests the idea).
Confirm: < 0.95726 (r25a).
