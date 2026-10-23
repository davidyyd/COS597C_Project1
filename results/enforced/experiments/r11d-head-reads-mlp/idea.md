# LM head reads the last MLP output undiluted
Parent: baseline
Type: explore

Hypothesis (from r10c): sublayers benefit from reading a fresh output re-normalized, apart from the long
residual sum. Test it at the output: the head reads norm(x_L) + beta norm(m_L) where m_L is the last MLP's
output (beta init 0), so the final computation is not drowned by the accumulated stream.
Confirm: < baseline by > 0.001. Refute: no gain (the effect is specific to MLPs reading attention).
Closest work: DenseFormer / MUDD (weighted sums of layer outputs), backout. Differs: re-normalized read of
only the last sublayer output, one scalar.
