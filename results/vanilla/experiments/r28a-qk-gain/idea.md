# Per-channel query gains in the QK-norm (learned RoPE-band weighting)
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: after RoPE + QK-norm, every channel pair contributes equally to the logit, but the pairs are
RoPE frequency bands: high-frequency pairs encode fine relative position, low-frequency pairs act as
nearly position-free content matching. A learned per-channel gain on the normalized query (shared by the
heads of a layer, fused into the RMSNorm weight) lets each layer weight positional vs. semantic bands.
The per-head temperature result (r01d) showed the logit scale matters; this is its per-band refinement.
Change: q <- rms_norm(q) ⊙ g (g ∈ R^128 per layer, init 1).
Confirm: < 0.95726 (r25a). Closest work: QK-norm with learnable gains (OLMo 2); here framed and tested as
frequency-band weighting.
