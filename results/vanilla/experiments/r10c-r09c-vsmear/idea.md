# r09c + per-channel smeared values
Parent: r09c-conv0-all. Type: exploit.

Hypothesis: smeared keys let a head find "the token after X"; smearing values (v_t + b ⊙ v_{t-1}) lets
the retrieved content also carry the preceding token, giving copy heads bigram content.
Confirm: < 0.9445 within the cap.
