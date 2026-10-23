# r38b with the per-head output gate replaced by a per-channel (elementwise) gate
Parent: r38b-r36a-ptr32 (with r38d's elementwise gate). Type: exploit.
The elementwise low-rank gate matched the headwise gate's gain on the control (-0.0053); a per-channel
gate can modulate individual value features. Same packed Muon-per-slice parameterisation (L x 512 x 32).
Confirm: < 0.933887 within the cap.
