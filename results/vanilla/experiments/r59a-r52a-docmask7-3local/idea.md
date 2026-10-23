# r52a with document masking only in the last global layer and 3 local heads (fit the cap with the MLP gate)
Parent: r52a-r51b-docmask-full. Type: exploit.
The doc-masked flex layers cost ~2x a windowed layer; masking only layer 7 and using 3 local heads (each
~1-2 s) may bring r52a (0.929378 @ 265 s) under the cap while keeping the MLP gate.
Confirm: < 0.930372 within the cap.
