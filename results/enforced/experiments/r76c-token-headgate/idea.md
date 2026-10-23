# Token-conditioned head gates (per-token, per-layer, per-head gate bias)
Parent: r75b-doc-ve-gate
Type: explore

Hypothesis: whether a head's retrieval is useful depends strongly on the current token (e.g. punctuation vs
content words use different heads). The head self-gate sees only what was retrieved; a per-token bias per
(layer, head) lets each token switch heads on or off directly. Cheap: 8192 x 32 table, an add on [B,T,H].
Change: g_h = a_h·y_h (+ low-rank) + B_l[token, h]; y_h *= 2σ(g_h). B init 0.
Confirm: below r75b by > 0.0005.
Closest work: token-dependent head gating / mixture-of-heads routing (MoH, 2024: router from the hidden state);
here a pure token-identity lookup added to a self-gate.
