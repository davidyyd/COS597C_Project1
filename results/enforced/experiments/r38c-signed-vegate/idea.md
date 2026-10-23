# Signed value-embedding gate
Parent: r37c-vesep-noouter
Type: exploit

With both value parts normalized, the VE gate 2 sigmoid(W x) in (0, 2) can only add token identity (and starts
at 1). A signed gate 2 tanh(W x) in (-2, 2), 0 at init, lets a head subtract the token's identity direction
from its value (e.g. to retrieve "what is different from this token") and starts without VE.
Confirm: < r37c (0.93796).
