# Attention output scales the MLP's write
Parent: baseline
Type: explore

Hypothesis: how much the MLP should write at a position depends on what attention just retrieved (e.g. a
confident retrieval of the next token makes the MLP's generic prediction less useful). Per-token scale on the
MLP output from the (bounded, normalized) attention output: x <- x + m * (1 + tanh(w . norm(a))), w init 0.
A multiplicative attention->MLP interaction (vs the additive read of r10c). The failed MLP self-gate (r03d)
used the unbounded MLP output as gate input; here it is unit-RMS.
Confirm: < baseline by > 0.001.
Closest work: GLU/SwiGLU (gate from the MLP input), highway gates. Differs: scalar gate from the attention
output of the same block.
