# Token shift on the attention input (q, k, v all see x_t and x_{t-1})
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: smeared keys (key-only, lag 1) helped; giving the whole attention input a learned per-channel
lag-1 mix lets queries ("what follows X?"), keys and values all carry bigram information at the cost of
one elementwise op per layer.
Change: x_att = x + a ⊙ x_{t-1} (a per channel, init 0) before the q/k/v projections.
Confirm: < 0.95726 (r25a). Closest work: RWKV time-mix token shift, smeared keys.
