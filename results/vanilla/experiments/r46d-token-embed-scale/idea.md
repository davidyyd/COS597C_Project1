# Per-token learned scale of the normalised input embedding
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: RMS-normalising the embedding forces every token's x0 (re-added before every layer) to the
same norm, including rare tokens whose embeddings are barely trained and noisy. A learned per-token
log-scale (one number per vocabulary entry, init 0) lets the model shrink unreliable tokens' re-injected
identity and amplify informative ones.
Change: x0 = norm(wte[t]) * exp(s[t]). Confirm: < 0.95726. Closest work: frequency-dependent embedding
scaling (adaptive input representations, Baevski & Auli 2019). Difference: learned per-token scalar on x0.
