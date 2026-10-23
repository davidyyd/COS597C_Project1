# Input gate on the value projection only
Parent: r58b-va-gates-adam
Type: explore

The attention input gate (all of q, k, v) gave -0.0028 (r64d). Role test: gate only the value projection's input
(values select their features; queries and keys see the ungated input). Compare with r64d (0.94343).
