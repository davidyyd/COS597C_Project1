# Document summary of the hidden state, injected mid-network
Parent: r25a-bf16-resid. Type: explore.
r77c (doc-mean of token embeddings into x0) gave -0.0030 on the control but hurt in the stack (r78b),
where it entered before the ctx-x0 conv and changed every layer's x0. Here the summary is of a
contextual state and enters once, mid-network: at layer 4, x += B norm(docmean(A norm(x))), A 512->128,
B 128->512 zero-init; docmean = causal running mean since the last BOS (cumsum difference).
Confirm: <= 0.9542 on control (then try in stack). Closest: uniform-weight linear attention / a single
"mean" head; TopicRNN. Difference: one doc-reset mean of a learned projection, no query.
