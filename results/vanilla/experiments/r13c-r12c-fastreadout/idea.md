# r12c with the multi-depth readout computed in one fused bf16 pass
Parent: r12c-packed-readout-ctxx0. Type: exploit.

Cost model from profiling: each elementwise pass over an fp32 (B,T,512) tensor costs ~0.05 ms. The
running-sum readout did ~3 fp32 passes per layer (fwd) plus backward. Here the readout sums the bf16
copies of the normalized MLP inputs (which autocast makes for the MLP matmul anyway) once at the end.
Nearly the same math (bf16 instead of fp32 readout terms). Confirm: ≈ 0.9413 and faster than 274 s.
