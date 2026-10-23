# Low-rank head gate -> temporal MLP input gate (layers 4-7)
Parent: r64a-ingate-instead
Type: exploit

Hypothesis: on the fast base the MLP input gate improved by -0.0021 when it also sees the previous position's input
(r71d): feature selection depends on local context. The cheap shifted-feature form failed on the best (r72b);
here the exact r71d form is used, paying for it by dropping the low-rank head self-gate.
Change: u <- u * 2σ(U (D u_t + D' u_{t-1})) in layers 4-7; lowrank head gate removed.
Confirm: below r64a (0.93169) and below r73a (same model without the temporal term).
Closest work: token shift (RWKV) / smear; here the shift drives only a feature-selection gate.
