# Document masking (in-graph block masks) on the bf16-rope baseline
Parent: r18a-bf16-rope. Type: explore (known method, efficient implementation as a possible base).

Measures the effect of document masking on the control model with the cheap in-graph mask construction.
Confirm: < 0.95739 at ~242 s or less.
