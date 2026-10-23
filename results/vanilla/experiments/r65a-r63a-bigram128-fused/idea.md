# r64c (rank-128 bigram logit prior) with the prior folded into the output-head matmul
Parent: r64c-r63a-bigram128 (0.927758 @ 263.6 s). Type: exploit (implementation; same model).
logits = [x, U[x_t]] @ [W_out, V]^T computes x W_out^T + U[x_t] V^T in one matmul, avoiding a separate
(B, T, 8192) logits addition (the 9 s cost of r64a/c). Confirm: ≈ 0.9278 within the cap.
