# MLP reads the token identity directly
Parent: baseline
Type: explore

Hypothesis: MLPs act as key-value memories keyed by local context; the token's own identity is the most
important key but after a few layers it is a small, diluted part of norm(x). Give each MLP a direct,
unnormalized-by-context copy: MLP input u = norm(x) + beta_l x0 (beta_l scalar, init 0). Unlike the
x0 re-injection into the residual (which also feeds attention and persists), this only changes what the
MLP reads.
Confirm: < baseline by > 0.001. Refute: no gain.
Closest work: x0/embedding skip (modded-nanogpt), DenseFormer. Differs: an MLP-only read of x0.
