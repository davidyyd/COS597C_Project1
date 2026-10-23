# Mid-network hashed bigram memory with a context gate (Engram-like)
Parent: r25a-bf16-resid. Type: explore.
A hashed (prev, cur) bigram table (16384 x 128, projected to 512, zero-init) injected into the residual at
layer 4 through a per-token gate read from the current state. Unlike the round-1 input-level hashed
bigram (no gain), the memory is read mid-network where the state can decide whether to use it.
Confirm: < 0.95726. Closest work: Engram (DeepSeek 2026, conditional n-gram memory), hashed n-gram
embeddings. Difference: a single small gated injection at mid depth.
