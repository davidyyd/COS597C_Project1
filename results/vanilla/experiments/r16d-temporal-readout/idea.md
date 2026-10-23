# Temporal readout: the head also sees the previous positions' final states
Parent: baseline. Type: explore.

Hypothesis: the final state at position t is the only input to the next-token prediction, but the final
states at t-1..t-3 also hold predictive information computed in parallel (e.g. what the model expected at
t-1, which is a strong prior for t+1 in multi-token words). A causal per-channel mix of the last 3 final
states into the head input gives the head this at almost no cost.
Change: x_out = norm(norm(x_t) + sum_{j=1..3} r_j ⊙ norm(x)_{t-j}), r init 0.
Confirm: < baseline. Refute: no gain (deep layers already aggregate this).
Closest work: depth readouts (DenseFormer), token shift (RWKV) — applied at the head over time.
