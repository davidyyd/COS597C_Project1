# Factorised VE in all layers + contextual (4-lag) conv on the 256-d table outputs
Parent: r73d-factve-all8 (0.953385). Type: explore.
Contextual value embeddings were a large win in full-table form; here the causal lag mix is applied to the
cheap 256-d factorised lookups before the up-projection. Confirm: < 0.953385.
