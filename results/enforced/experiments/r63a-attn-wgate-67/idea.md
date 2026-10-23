# Best + attention write gates in layers 6-7
Parent: r61a-lrodd-outgate-late (0.93223)
Type: exploit

Attention write gates (from the attention input) partially add to MLP write gates on the fast base (r62d).
Layers 4-7 did not fit the step cap; try the last two layers. Confirm: < r61a.
