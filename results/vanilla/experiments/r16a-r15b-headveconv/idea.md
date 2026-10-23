# r15b with per-head contextual value-embedding conv weights
Parent: r15b-r14a-noreadout (adds r15c's per-head VE conv). Type: exploit.

Per-head lag weights for the contextual VE were better than per-channel in r15c (0.9385 vs 0.9389).
On the fast r15b (251.8 s) this should give the best valid model.
Confirm: < 0.9397 within the cap.
