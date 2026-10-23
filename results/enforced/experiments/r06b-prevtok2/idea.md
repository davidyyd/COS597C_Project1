# Keys see the two previous tokens
Parent: r05b-best-x0gate
Type: exploit

Prev-token keys (k input += lambda ⊙ x0_{t-1}) helped. Extend to a trigram context:
k input += lambda_l ⊙ (x0_{t-1} + mu ⊙ x0_{t-2}), mu a single per-channel vector shared by all layers (init 0; one global mix to stay under the step-time cap). Lets heads match longer contexts.
Confirm: < r05b by > 0.0005.
