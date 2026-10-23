# Undiluted reads without the extra norms
Parent: r14c-chain3-attn-reads-attn
Type: exploit (simplification / time)

The reads use norm(a), norm(m). Use the raw outputs instead (u = norm(x) + beta a + ...): the sublayer
outputs' scale is already set by their projections, and dropping two norms per block saves time.
Confirm: within +0.0005 of r14c. Refute: worse (per-token normalization of the read matters).
