# Depth gates read from the previous MLP's normalized input (no extra read of the residual)
Parent: r03a-depthgates-temp. Type: exploit.

Hypothesis: r03a's gates (0.9496) need context (r04a: identity-only gates lose most of the gain), but
reading them from the residual costs ~3 ms/step (over the time cap). The normalized MLP input of the
previous block is already computed and is almost the same state; reading the gates of layer i from it
(layer 0 from x0) should keep the gain at almost no cost.

Change: block returns h = norm(x) (its MLP input); layer i's gates g = 2 sigmoid(h_{i-1}[:32] W_i).
Confirm: ≈ 0.9496 at < 262 s.
