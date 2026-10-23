# Time-shifted undiluted read
Parent: r50c-compA-fullchain
Type: explore

The read chain gives sublayers the previous sublayer's output at the same position. Add a read of the previous
block's MLP output at the previous position (attention input += gamma norm(m_{l-1})[t-1]): local context
(what the model concluded about the previous token) without going through attention. Confirm: < r50c.
Closest work: token shift (RWKV), smear (modded-nanogpt: shifts the embedding). Differs: a shifted read of
the previous layer's normalized MLP output into the attention input.
