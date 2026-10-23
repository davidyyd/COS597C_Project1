# Learned per-head log-length query scaling
Parent: baseline. Type: explore.

Hypothesis: the number of visible keys grows with position t (up to 1024/2048), so with a fixed
temperature the softmax mass on the relevant key decays with log t; QK-norm caps logits, making this
worse. Scaling queries by (1 + a_h log(1+t)) with learned a_h lets each head keep its sharpness
independent of context length (or choose not to).

Change: q_t <- (1 + a_h log(1+t)) norm(q_t), a_h init 0.
Confirm: < baseline. Refute: no gain (the static per-head temperature of r01d already covers it).
Closest work: Scalable-Softmax (SSMax, Nakanishi 2025), log-n attention scaling (Chiang & Cholak 2022).
Difference: learned per-head slope (init 0) on top of QK-norm.
