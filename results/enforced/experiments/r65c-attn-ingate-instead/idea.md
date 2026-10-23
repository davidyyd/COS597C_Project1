# Best: attention input gates instead of MLP input gates (layers 4-7)
Parent: r64a-ingate-instead (0.93169)
Type: exploit (time-neutral swap; adding attention input gates on top measured ~70 ms)

The attention input gate (u <- u * 2 sigmoid(U V u), rank 32, AdamW) was the strongest input gate on the fast base
(-0.0028). r64a has no step-time headroom, so swap it in for another component. Confirm: < r64a.
