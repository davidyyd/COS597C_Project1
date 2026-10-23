# Best model: swap the half-layer low-rank gate for a novelty term in the output temperature
Parent: r28a-best-outtemp
Type: exploit

logits *= 1 + tanh(w . norm(x_L) + b nov). Half-layer low-rank gate removed for time (see r32a).
Confirm: < r28a.
