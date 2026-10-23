# Self-gated MLP output
Parent: baseline
Type: explore

Hypothesis: the self-gate on attention heads (r01b, -0.0024) may be a generic benefit of letting a block
damp its own write based on what it computed, not specific to attention. Test the same on the MLP:
out <- out * 2 sigmoid(c . out), c in R^512 per layer, init 0.
Confirm: < baseline by > 0.001 means generic; no gain means the effect is attention-specific (softmax's
inability to output nothing).
Closest work: gated attention (Qiu 2025) / GLU; here a scalar per-token gate from the block's own output.
