# Baseline with bf16 RoPE tables (attention runs in bf16 in every layer)
Parent: baseline. Type: explore (precision/efficiency control).

Finding: in the baseline the fp32 RoPE tables promote q,k to fp32, so the six flex-attention (sliding
window) layers run in fp32 (and VE layers have fp32 values), while the two SDPA layers are cast to bf16 by
autocast. The original nanochat model used bf16 tables. This run measures what bf16 tables (and a bf16
value-embedding add) do to speed and quality, so later models can be judged fairly. Not an
architectural idea; a control that I disclose as such.
Confirm/refute: val within ~0.001 of baseline and faster => attention precision was an accident.
