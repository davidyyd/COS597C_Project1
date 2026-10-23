# Hard-wired induction prior in the global layers (v3: lowering-safe score_mod form)
Parent: r57d-induction-prior-v2. Type: explore. Same hypothesis; written as where(match, score + beta, score),
the form that lowers correctly. Confirm: < 0.95726 (r25a).
