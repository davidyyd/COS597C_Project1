# Multi-depth readout: the output head reads every layer's normalized state
Parent: baseline. Type: explore.

Hypothesis: the head only sees the final residual; intermediate representations (already normalized
as MLP inputs) may hold information the last layers overwrite. A learned weighted sum of all depths
into the head is a cheap shortcut (no extra norm: reuses each block's MLP input norm).
Change: x_out = norm(norm(x_L) + sum_i beta_i h_i), h_i = block i's MLP input, beta init 0.
Confirm: < baseline. Refute: no gain.
Closest work: DenseFormer (depth-weighted averaging), ELMo scalar mix. Difference: readout-only, reusing
the normalized MLP inputs.
