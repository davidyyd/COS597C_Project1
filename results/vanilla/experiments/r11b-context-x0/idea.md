# Contextual x0: causal per-channel convolution over the token embeddings
Parent: baseline. Type: explore.

Hypothesis: the embedding x0 is re-injected before every block (x0 lambdas), so whatever x0 contains is
available everywhere. The conv-layer-0 result (r08d) shows local mixing early is valuable; putting a
short causal conv into x0 itself makes local context (previous 3 tokens) available to every layer.

Change: x0 = norm(sum_{j<4} w_j ⊙ wte(t_{i-j})), w per channel (init w_0 = 1, others 0).
Confirm: < baseline. Refute: no gain.
Closest work: modded-nanogpt smear gate (previous-token embedding mixed in with a gate), RWKV token shift,
Canon layers. Difference: per-channel width-4 conv on the re-injected embedding.
