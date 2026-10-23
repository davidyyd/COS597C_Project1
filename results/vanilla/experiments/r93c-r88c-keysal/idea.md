# r88c + multiplicative per-key salience (shares the smear-gate matmul)
Parent: r88c-r87b-dist-nometa. Type: exploit (combine with the r92d explore: -0.0021 on the control).
k_s = norm(k_s) * 2 sigmoid(w_h . x_s[:32]); the salience logits come from 4 extra columns of the existing
selective-smear gate matmul. Confirm: < 0.923063 within the cap.
