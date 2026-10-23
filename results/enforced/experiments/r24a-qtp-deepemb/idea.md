# Deep token embedding on r23c-cheap-qtemp-prevtok
Parent: r23c-cheap-qtemp-prevtok
Type: exploit

Add the token-wise MLP refinement of the embedding (x0 = norm(e + MLP(e)), hidden 512, zero-init output;
r23d: -0.0039 on baseline) to this parent; the refined x0 feeds the re-injection, x0 gate and reads.
Confirm: < parent by > 0.001.
