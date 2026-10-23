# r79a with the grouped MLP hidden gate restored in layers 4-7 only
Parent: r79a-r78a-dropve3. Type: exploit.
The full MLP gate (all 8 layers) was worth ~6e-4 for ~3.2 s; r79a has 2.9 s headroom. Restoring it in
the second half costs ~1.6 s; tests whether its value is concentrated in late layers. Confirm: < 0.924342
within the cap.
