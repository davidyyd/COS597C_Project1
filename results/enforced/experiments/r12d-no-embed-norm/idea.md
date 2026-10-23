# Token embeddings keep their magnitude (no input RMSNorm)
Parent: baseline
Type: explore

Hypothesis: the baseline normalizes every token embedding to unit RMS, erasing a per-token magnitude that
could encode salience (rare/content tokens vs. frequent function tokens) and set how strongly x0 is
re-injected per token. Restoring magnitude channels erased by gainless norms helped elsewhere (query
temperature, output temperature, x0 gate). Change: x0 = wte[idx] (init N(0,1), so RMS ~1 at start).
Confirm: < baseline by > 0.001. Refute: no gain / instability.
Closest work: most GPTs do not normalize embeddings (GPT-2); modded-nanogpt/nanochat do. Differs from both
only in removing this one norm in a model with x0 re-injection.
