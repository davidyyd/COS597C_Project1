# Per-dimension query feature gate
Parent: r09a-gate-multiscale
Type: exploit

Query-side controls helped (temperature). Extend from one scalar to a per-dimension gate: after QK-norm and
the temperature, q_h <- q_h * 2 sigmoid(U_h V x) (V [16,512] shared, U_h [128,16] per head; rank 16 for the param cap, U init 0), so a
token chooses which features (and rotary frequencies) it matches on. Confirm: < r09a by > 0.0005.
Closest work: learned QK-norm gains (static per channel); here per token.
