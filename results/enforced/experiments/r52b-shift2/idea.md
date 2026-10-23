# Two time-shifted reads into attention
Parent: r51d-c-shift-read
Type: explore

Attention also reads the previous block's attention output at the previous position (in addition to its MLP
output at t-1). Confirm: < r51d (0.93647).
