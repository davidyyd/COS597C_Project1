# Per-token adaptive logit soft-cap
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: a fixed cap (15) limits confident predictions everywhere equally; easy tokens want a high cap
(sharp), uncertain ones a low cap. c_t = 15 * 2 sigmoid(w . x_t[:32]) (zero-init => 15),
logits = c_t tanh(l / c_t). Confirm: < 0.95726. Closest: Gemma-2 fixed soft-capping; learned temperature
scaling. Difference: a per-token cap read from the final state.
