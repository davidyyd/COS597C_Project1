# Key-side value gate: each token scales the value it offers
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: the output gates (query side) decide how much a head writes; nothing decides how much each
token *offers* as a value (function words, punctuation, BOS sinks could offer little). A per-token,
per-head scalar on the values, read from 32 channels of the key token (init neutral), lets the content be
weighted at the source, independently of the attention weights (which must sum to one).
Change: v_s <- 2 sigmoid(w_h · x_s[:32]) v_s. Confirm: < 0.95726.
