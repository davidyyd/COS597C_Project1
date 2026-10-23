# Differential attention in the two full-attention layers
Parent: r25a-bf16-resid. Type: explore (known method, applied only to the global layers).
Hypothesis: global layers attend over up to 2048 keys, where softmax noise (attention to irrelevant
context) is largest; differential attention (difference of two softmaxes over half-dim q/k, learned
lambda, per-head norm) cancels common-mode noise. Applied only to the global layers, cost stays small.
Confirm: < 0.95726. Closest work: Differential Transformer (Ye et al. 2024).
