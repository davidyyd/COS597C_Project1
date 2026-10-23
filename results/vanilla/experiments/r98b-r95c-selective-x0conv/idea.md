# Selective ctx-x0 conv: per-token lag gates on the input-embedding conv
Parent: r95c-r94b-gatesall-ve3. Type: exploit (transfer the selective-lag idea that made ctx-VE work).
The VE lag gates are worth 1.1e-3 (r83c) while the static ctx-x0 conv is worth only 2.7e-4 (r83b). Give the x0
conv the same selectivity: lag j of channel quarter c is scaled by 2 sigmoid(w_{j,c} . norm(e_t)[:32]).
Confirm: < 0.921402.
