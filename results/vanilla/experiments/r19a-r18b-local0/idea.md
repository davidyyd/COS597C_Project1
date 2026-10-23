# r18b with layer 0 as local attention (window 128) instead of the value conv
Parent: r18b-r17a-bf16. Type: exploit.

On the baseline, local attention in layers 0-1 (r18d, -0.0050) beat the conv layer 0 (r08d, -0.0035).
Tests whether content-based local mixing in layer 0 also beats the fixed conv in the full stack.
Confirm: < 0.9369 (time may exceed the cap: the conv layer was cheaper).
