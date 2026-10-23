# Self-value skip in attention
Parent: r01b-attn-selfgate (baseline version measured 69.4 ms)
Type: explore

Hypothesis: a head's own token is one key among many; to pass its own value through, it must win the softmax,
which competes with retrieval. Add y_h <- y_h + g_h v_h(t) (per-head scalar g, init 0) before the self-gate: a
free path for the token's own value, so attention can spend its probability mass on other positions.
Confirm: < r01b (0.95653) by > 0.001.
Closest work: exclusive self attention (XSA: removes the self component), value residual (ResFormer). Differs:
adds the same position's value of this layer with a learned per-head weight.
