# Non-linear (self-gated) values
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: attention outputs are averages of linear projections of the input; a cheap per-dimension
non-linearity on the values before averaging (v <- v * 2 sigmoid(s_h v), s per head init 0 = identity) lets each
value dimension saturate/suppress itself, so the average is of non-linear features (more expressive mixing,
no extra matmul). Complements the output self-gate (non-linearity after averaging).
Confirm: < r01b (0.95653) by > 0.001.
Closest work: SiLU/GLU (MLPs), gated attention (after averaging). Differs: elementwise self-gating of values
before attention, one scalar per head.
