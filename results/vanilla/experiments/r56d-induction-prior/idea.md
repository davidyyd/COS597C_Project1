# Hard-wired induction prior in the global layers
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: induction heads ("find the earlier occurrence of my token and attend to what followed it")
are learned slowly; in the two global layers, add a fixed per-head bonus beta_h * [token_{s-1} == token_t]
to the attention score (beta = 0, 1, 2, 4 for the 4 heads), so some heads start with an exact induction
bias. Uses only tokens <= t (causal). No learnable parameters in the score_mod (fast).
Confirm: < 0.95726 (r25a). Closest work: induction heads (Olsson et al. 2022); n-gram/induction priors.
