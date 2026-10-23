# r17a (local attention in layer 1) with bf16 RoPE tables
Parent: r17a-r16a-localattn1. Type: exploit.

r17a is the best model (0.9370) but 266.5 s. With bf16 RoPE tables (as in the original nanochat; see
r18a) attention runs in bf16 in all layers; this should bring it under the cap.
Confirm: ≈ 0.937 at < 261.7 s.
