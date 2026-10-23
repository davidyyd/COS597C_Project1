# r18b + contextual value embeddings also added to the keys
Parent: r18b-r17a-bf16. Type: exploit.

Hypothesis: value embeddings give attention outputs exact token identity, but the keys used for matching
are computed from the drifting residual. Adding the same (contextual, i.e. n-gram-like) value embedding to
the keys (per-head weight, init 0, no new tables) lets heads match on exact local n-gram identity, which
is what induction/copy heads need. Cost: one add per VE layer.
Change: k <- k + a_h * ve (before RoPE), a_h per head, init 0.
Confirm: < 0.9369 within the cap.
