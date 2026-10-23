# r16b without the head output gate
Parent: r16b-r15b-attn1. Type: exploit.

r16b (layer-1 attention back, 0.9381) is over the cap by ~8 s. Tests whether the head gate (cheap
alone, -0.0054 on the baseline) is still worth its cost in the full stack, or layer-1 attention is the
better use of the time budget.
Confirm: < 0.9391 within the cap.
