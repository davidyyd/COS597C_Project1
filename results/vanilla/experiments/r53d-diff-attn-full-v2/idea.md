# Differential attention in the two full-attention layers (retry of r52b)
Parent: r52b-diff-attn-full (crashed: lambda parameters of the non-full layers had no gradient; now only
the full layers own one). Type: explore (known method, applied only to the global layers).
Same hypothesis as r52b. Confirm: < 0.95726 (r25a).
