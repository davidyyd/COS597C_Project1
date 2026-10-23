# Normalised residual stream: rescale to unit RMS after every block (learned per-layer scale)
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: in pre-norm transformers the residual norm grows with depth, so later blocks' updates are
relatively smaller and early features dominate the stream. Renormalising the stream after each block
(x <- s_i * norm(x)) keeps every block's contribution on the same scale, as in normalised transformers.
Confirm: < 0.95726. Closest work: nGPT (Loshchilov et al. 2024, hypersphere residuals), Mix-LN/Peri-LN.
Difference: a single post-block renormalisation with x0 re-injection kept.
