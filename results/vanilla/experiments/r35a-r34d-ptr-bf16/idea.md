# r34d with the pointer scores kept in bf16 (fp32 only inside the fused reductions)
Parent: r34d-r32a-pointer-dense. Type: exploit.

r34d (r32a + pointer) reached 0.934204 but at 265 s. The (B,T,T) pointer scores were materialised in
fp32; keeping them bf16 (logsumexp still accumulates in fp32) halves that memory traffic.
Confirm: ≈ 0.9342 at < 261.7 s.
