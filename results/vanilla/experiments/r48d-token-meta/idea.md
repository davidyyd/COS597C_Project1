# Token metadata embedding (byte length, leading whitespace, first-character class)
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: orthographic properties of a token (how many bytes, whether it starts a word, whether it is
capitalised / numeric / punctuation / non-ASCII) are shared across many tokens and predictive (e.g. a
capitalised word-start after a period), but each token embedding must learn them separately, poorly for
rare tokens. A small embedding of these tokenizer-derived categories (192 combinations) added to the token
embedding gives them directly. Computed from the tokenizer at start-up; no new information beyond the token id.
Confirm: < 0.95726 (r25a). Closest work: factored/feature-rich embeddings (e.g. character-aware embeddings).
