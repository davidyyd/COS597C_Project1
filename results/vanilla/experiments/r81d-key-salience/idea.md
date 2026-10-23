# Key salience bias carried in one q/k channel
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: some tokens are worth attending to regardless of the query (content words, delimiters); a
query-independent per-key score bias lets every head express this without spending q.k capacity.
Change: after QK-norm, the last channel of q is set to sqrt(D) and the last channel of k to b_s = w_h . x_s
(zero-init), so score_ts += b_s exactly; works with SDPA and flex unchanged.
Confirm: < 0.95726. Closest: attention sinks/biases, Selective Attention (Leviathan et al.), FoX forget gates
(distance-dependent); here a static per-key additive salience via channel augmentation.
