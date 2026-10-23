# Decaying-sum undiluted read on the cheap lineage
Parent: r21b-cheap-lineage-gate
Type: exploit

Generalize the pairwise undiluted reads: keep r, a decaying sum of normalized sublayer outputs
(r <- gamma_j r + norm(out_j), gamma init 0) and let every sublayer read norm(x) + beta_j r. With gamma = 0
it is "read the previous sublayer only"; gamma > 0 reaches older outputs. One read per sublayer.
Compare with the pairwise chain on the same parent (r21c). Confirm: <= r21c.
Closest work: DenseFormer / Attention Residuals (learned weights over all earlier outputs). Differs: one
exponentially decaying, normalized sum, read additively, 4 scalars per block.
