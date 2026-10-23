# Self-gated attention heads (gate from the head's own output)
Parent: baseline
Type: explore

Hypothesis: softmax attention must always output a convex combination of values, so a head cannot say
"nothing relevant here". Gated attention fixes this with a gate computed from the input x. Here the gate is
computed from what the head retrieved: a head can switch itself off depending on its own output, a
nonlinearity on the attention output (like a GLU on the retrieved value).

Change: y_h <- y_h * 2 sigmoid(a_h . y_h), a_h in R^128 per head, init 0 (gate = 1 at init).

Confirm: val_loss < baseline by > 0.002. Refute: no gain.
Closest work: Gated attention (Qiu et al. 2025, gate = sigmoid(W x)), attention sinks. Differs: gate is a
function of the attention output itself (self-gating), 128 parameters per head.
