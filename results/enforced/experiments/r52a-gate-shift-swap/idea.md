# Best (r51b): swap the attention<-previous-attention read for the time-shifted MLP read
Parent: r51b-c-gate-halflr (0.93611)
Type: exploit (time-neutral swap; adding the shift read measured 69.4 ms)

Attention reads norm(m_{l-1}) at t and at t-1 instead of norm(m_{l-1}) and norm(a_{l-1}) at t. Confirm: < r51b.
