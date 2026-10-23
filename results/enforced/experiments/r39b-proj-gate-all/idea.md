# Per-token value loudness in all layers
Parent: r38a-gated-proj
Type: exploit

In r38a the projected-value gate g1 exists only in VE layers. Add it in the others too: v = s g1 norm(W_v u),
g1 = 2 sigmoid(W x) (init 1), i.e. a per-token, per-head value loudness (unnormalized per-token value scales
hurt in r06a; with normalized values the gate is the only magnitude). Confirm: < r38a.
