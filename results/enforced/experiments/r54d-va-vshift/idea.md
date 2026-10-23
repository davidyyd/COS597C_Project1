# Previous position's value as a normalized value source
Parent: r43a-base-valuepath
Type: explore

Values get the previous position's (normalized) value with a signed per-head gate (0 at init): v_t <- v_t +
tanh(W x) norm(v_{t-1}), so an attended position also reports its local context. Confirm: < r43a (0.94690).
Closest work: key offset / smear (shift on keys or embeddings), RWKV token shift. Differs: shift on values with
normalized gated mixing.
