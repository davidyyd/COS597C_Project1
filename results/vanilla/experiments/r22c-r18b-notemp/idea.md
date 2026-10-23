# Ablation of r18b: remove the per-head attention temperature
Parent: r18b-r17a-bf16. Type: exploit (ablation).

r18b (0.9369 at 261.0 s) uses the whole time budget, and several later additions gave small gains at
4-10 s each. This ablation measures what the per-head attention temperature is worth inside the full stack (val and seconds),
to decide whether its time is better spent on another component (local attention in layer 0: -0.0006/+10 s,
value smear: -0.0006/+7.6 s).
