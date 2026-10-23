# r44c with selective (token-gated) contextual value-embedding lags
Parent: r44c-r42b-unet. Type: exploit.
Selective (token-gated) smears were a free gain (0.0008); the ctx-VE lags are static per head. A per-token
gate per lag (read from 32 channels of x0, init neutral) lets each token decide how much predecessor
identity its value embedding carries. Confirm: < 0.932038 within the cap.
