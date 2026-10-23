# Learned per-head sharpness scale on the doc temperature (cheap lineage)
Parent: r21c-cheap-gate-chain
Type: exploit

The doc temperature 1 + tanh(c_h f_doc) is bounded to (0, 2). Multiply by a learned per-head scale s_h
(init 1) so heads can choose their overall sharpness: q *= s_h (1 + tanh(c_h f_doc)).
Confirm: < r21c (0.94791) by > 0.0005.
Closest work: learned attention scale / QK-norm gains; here combined with the doc-position term.
