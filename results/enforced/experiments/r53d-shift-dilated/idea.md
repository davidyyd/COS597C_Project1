# Dilated time-shifted read (t-1 and t-2)
Parent: r51d-c-shift-read
Type: explore

Extend the time-shifted read to two positions back (the previous block's MLP output at t-1 and t-2): a cheap
short causal convolution over the previous layer's conclusions. Confirm: < r51d (0.93647).
