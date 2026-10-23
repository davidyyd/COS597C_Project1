# Self-gated heads + bounded per-token query temperature
Parent: r01b-attn-selfgate
Type: exploit (retry of the diverged r01c with a bounded form)

Hypothesis: QK-norm forces equal attention sharpness for all tokens; a per-token, per-head temperature helps.
r01c used exp(W x) which diverged at step 458. Bounded version: q_h <- norm(q_h) * (1 + tanh(w_h . x)),
range (0, 2), W init 0. Confirm: < r01b by > 0.001 and stable.
Closest work: learned QK-norm gains (static), scalable softmax (position-based), selective temperature.
