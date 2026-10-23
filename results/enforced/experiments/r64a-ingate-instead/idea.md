# Best with MLP input gates instead of MLP write gates (layers 4-7)
Parent: r61a-lrodd-outgate-late (0.93223 / repeat 0.93208)
Type: exploit (time-neutral comparison)

On the fast base the input-side gate (feature selection: u <- u * 2 sigmoid(U V u)) was slightly better than the
write gate (r63c -0.0018 vs r60b -0.0016). Swap. Confirm: < r61a.
