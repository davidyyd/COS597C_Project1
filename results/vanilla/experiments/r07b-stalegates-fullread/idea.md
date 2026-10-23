# Stale depth gates read from the full normalized state
Parent: r05a-stale-gates. Type: exploit.

Hypothesis: the depth gates read only 32 of 512 channels (an arbitrary inherited choice from the VE
gate). Gates need context (r04a), so a full-width read should give better gating decisions; the read is
from the already-computed bf16 MLP input, so it should stay cheap.

Change: g_i = 2 sigmoid(h_{i-1} W_i), W_i in R^{512x2} (was 32x2).
Confirm: < 0.9505 at ~baseline speed. Refute: no gain (32 channels suffice: the gate is low-information).
