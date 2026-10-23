# r15b + contextual values: per-head 4-lag causal conv on the attention values
Parent: r15b-r14a-noreadout. Type: exploit.

Contextual VE (lagged value embeddings) was the strongest gain; the projected values themselves can also
be made contextual: v_t <- v_t + sum_{j=1..4} c_{j,h} v_{t-j}, with per-head lag weights (16 numbers per
layer, packed; cheap gradient). Generalises value smear (r10c, per-channel lag 1, -0.001 but slow).
Confirm: < 0.9397 within the cap.
