# r42a + memory diagnostics (prints allocated/peak memory around forward/backward at steps 1-2)

Parent: r42a-d16m6-gate-fused. Diagnostic: the head gate costs +3.6GB peak and d17 versions OOM; print torch.cuda.memory_allocated / max_memory_allocated after each forward/backward of steps 1-2 to locate the peak. Training otherwise identical (also a seed-identical replicate of r42a).
