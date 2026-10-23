# Swap the deep embedding for prev-token keys
Parent: r28a-best-outtemp
Type: exploit

Remove the deep embedding (~1.5 ms) and add the prev-token key input instead (both give the model cheap
local token features). Compare with r30a (no deep embedding). Confirm: < r30a.
