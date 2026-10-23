# Packed r10c + log-n query scaling + contextual x0
Parent: r12a-packed-r10c (adds r09d-logn-temp and r11b-context-x0). Type: exploit.

Hypothesis: contextual x0 (-0.0057 alone) and log-n query scaling (-0.0038 alone) were both cheap in
isolation and address different weaknesses (local context in the re-injected embedding; sharpness
vs. number of keys); with packing they now fit.
Confirm: < r12a.
