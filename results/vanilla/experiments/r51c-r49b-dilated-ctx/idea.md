# r49b + dilated contextual x0 lags (4, 8, 12)
Parent: r49b-lean-tokmeta-local0. Type: exploit.
The contextual x0 covers lags 0-3; three dilated lags give a cheap view of the last dozen tokens in the
re-injected embedding (computed once at the input in bf16). Confirm: < 0.931007 within the cap.
