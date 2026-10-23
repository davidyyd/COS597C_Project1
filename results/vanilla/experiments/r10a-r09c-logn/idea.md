# r09c + learned per-head log-length query scaling
Parent: r09c-conv0-all (adds r09d-logn-temp). Type: exploit.

Hypothesis: log-n query scaling (-0.0038 alone, free) keeps sharpness as the number of visible keys grows,
which the static temperature in r09c does not; it should add.
Confirm: < 0.9445 within the cap.
