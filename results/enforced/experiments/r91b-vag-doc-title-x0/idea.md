# Late layers re-inject the document's first token (title signal)
Parent: r83a-vag-dm-ingate
Type: explore
Hypothesis: the first token after a document's BOS (often a title or source word) is a strong, cheap document-level
cue; late layers must otherwise retrieve it by attention. Re-inject its normalized embedding (like the x0
re-injection) in layers 4-7 with a learned per-layer weight (zero-init). At the BOS itself the token's own
embedding is used (causal).
Confirm: below r83a (0.94006) by > 0.0005.
