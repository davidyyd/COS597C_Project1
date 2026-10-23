# Stale depth gates + temperature + per-head scalar smeared keys
Parent: r05a-stale-gates (adds a cheaper version of r02d-smeared-keys). Type: exploit.

r06b (per-channel smear) reached 0.9476 but 267 s. A single smear weight per head (4 instead of 512
numbers per layer) tests whether channel selectivity matters and may be cheaper.
Confirm: close to 0.9476 within the cap.
