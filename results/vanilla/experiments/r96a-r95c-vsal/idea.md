# r95c + per-key value salience (4 more columns of the shared smear/salience gate)
Parent: r95c-r94b-gatesall-ve3 (0.921402 @ 255.2 s). Type: exploit.
r95d: key + value salience are additive on the control (0.952950, -0.0043; key alone -0.0021). In the stack the
key salience already sits in the smear-gate matmul; add v_s *= 2 sigmoid(w_h . x_s[:32]) from 4 more columns.
Confirm: < 0.921402 within the cap.
