# Gate the projected value part too
Parent: r37c-vesep-noouter
Type: exploit

r37c: v = s (norm(W_v u) + g norm(ve)) in VE layers. Give the projected part its own gate:
v = s (g1 norm(W_v u) + g2 norm(ve)), g1 = 2 sigmoid(W1 x) (init 1): each head chooses per token how much context
vs token identity its value carries (and how loud). Confirm: < r37c (0.93796).
