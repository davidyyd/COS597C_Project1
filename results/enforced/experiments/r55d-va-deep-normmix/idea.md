# Deep embedding with normalized gated mixing
Parent: r49b-va-deepemb (0.94410)
Type: explore

Apply the value-path principle to the deep embedding: x0 = norm(e + tanh(w . e) norm(MLP(e))) — the refinement is
unit-scale and its weight is a per-token gate (0 at init), instead of an unnormalized residual add.
Confirm: < r49b.
