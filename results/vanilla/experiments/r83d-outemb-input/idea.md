# Output embedding of the current token as an extra input feature
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: the untied head row W_out[x_t] encodes how token x_t is predicted (its distributional
"as a target" role), information the input embedding learns separately and slowly. Feeding it to the
input (x = norm(wte[x_t] + lambda * W_out[x_t]), lambda per channel, zero-init; gradients flow into both)
shares the two views without tying them. Confirm: < 0.95726. Closest: weight tying (Press & Wolf),
which forces wte = W_out; here both stay free and the output view is added as a feature.
