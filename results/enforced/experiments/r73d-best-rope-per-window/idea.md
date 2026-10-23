# RoPE base matched to each multi-scale head's window
Parent: r64a-ingate-instead
Type: explore

Hypothesis: in the sliding layers the four heads see windows 128/256/512/1024, but all use RoPE base 10000 (designed
for long contexts). For a head that sees only 128 tokens, most rotary channels barely rotate within its window,
so its positional resolution is wasted. Scaling the base with the receptive field gives each head the same
spectrum relative to what it can see.

Change: in the multi-scale (sliding) layers, head h with window w_h uses RoPE base 10000 · w_h / 1024
(1250, 2500, 5000, 10000); full-attention layers unchanged. No parameters, no extra compute.

Confirm: below r64a (0.93169) by > 0.0005. Refute: no change (the low-frequency channels act as useful
position-free channels).

Closest work: per-head / per-layer RoPE bases (e.g. variable-base RoPE studies, NTK scaling for context
extension), multi-scale windows (Gemma local/global layers use different bases per LAYER type). Here the base is
set per HEAD from that head's own window.
