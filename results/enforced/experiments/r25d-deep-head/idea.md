# Deep head: token-wise MLP before the LM head
Parent: baseline
Type: explore

Hypothesis: the deep embedding (r23d, -0.0039) helped because the re-injected x0 became non-linear token
features. Symmetrically, the head is a single linear map of norm(x_L); a small token-wise MLP refining the
final state (norm(x + MLP(x)), hidden 512, zero-init output) gives a non-linear readout specialized for the
vocabulary projection, without attention.
Confirm: < baseline by > 0.001. Compare with r23d.
Closest work: deeper prediction heads (e.g. MLP heads in multi-token prediction), deep embedding (r23d).
