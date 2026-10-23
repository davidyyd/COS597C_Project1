# Query-gated position sensitivity (partial RoPE with a per-token balance)
Parent: baseline
Type: explore

Hypothesis: some queries should match mostly by relative position (local syntax), others by content
regardless of distance (retrieval). With partial RoPE (half the rotary pairs unrotated), each query can
choose per head: scale its rotary dims by (1 + g) and its position-free dims by (1 - g), g = tanh(w_h . x)
(init 0). Query-side controls worked well (temperature); this one controls position vs content.
Confirm: < baseline by > 0.001. (Partial RoPE alone is known; the per-token gate is the new part.)
Closest work: partial / truncated RoPE (GPT-J, modded-nanogpt), NoPE layers. Differs: per-token,
per-head balance between rotary and position-free channels.
