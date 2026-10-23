# Half-width value embeddings in every layer
Parent: r39d-base-gated-values
Type: explore

Hypothesis: with the value path fixed, token identity in values is very useful, but only 4 of 8 layers get it.
Same embedding budget, spread out: 8 tables of width 256 (instead of 4 of width 512); in every layer two of the
four heads mix a value embedding into their values (even layers heads 0-1, odd layers heads 2-3), with the same
gated normalized mixing. Confirm: < r39d (0.94631) by > 0.001.
Closest work: value embeddings (modded-nanogpt, alternate layers, full width). Differs: half-width tables in
every layer, per-head assignment.
