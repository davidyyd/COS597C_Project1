# Multiplicative undiluted read
Parent: r10c-mlp-reads-attn
Type: explore

The MLP's additive read of the fresh attention output (u = norm(x) + beta norm(a); r10c, -0.0052) is the
strongest single mechanism found. Test a multiplicative version: u = norm(x) * (1 + beta norm(a)), where the
retrieval modulates the MLP input channel-wise (a bilinear interaction the ReLU^2 MLP cannot form from a sum
in one layer). Same cost. Confirm: < r10c (0.95379).
Closest work: GLU / bilinear layers (gate from the same input); FiLM (conditioning by another stream).
Differs: the block's own attention output modulates its MLP input.
