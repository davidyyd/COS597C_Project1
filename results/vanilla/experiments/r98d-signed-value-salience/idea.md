# Signed value salience: keys can contribute negatively
Parent: r95d-key-value-salience. Type: explore.
r95d scales each key's value by 2 sigmoid(.) in (0, 2). Softmax attention can only add convex combinations of
values; a signed per-key weight 1 + 2 tanh(w . x_s[:32]) in (-1, 3) (init 1) lets a token inhibit (subtract)
its value, like the negative branch of differential attention but per key and with no extra attention map.
Confirm: < 0.952950. Closest: Differential Transformer (difference of two softmaxes); ours is per key.
