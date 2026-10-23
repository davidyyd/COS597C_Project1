# r26b (bf16 AdamW, now compiled) with LR 1.2e-2

Parent: r26b-d21-adambf16 (best, 0.77482; = r25a + AdamWBF16). AdamWBF16 update now torch.compiled with lr/bias-corrections as tensors (eager took 28 ms/step). Exploit: with Muon mu 0.9, LR 1.2e-2 beat 1e-2 at d16 (r26d); d21 still at 8e-3 -> try 1.2e-2.
