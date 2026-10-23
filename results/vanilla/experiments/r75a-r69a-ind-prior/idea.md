# r69a with an induction-candidate logit prior (prior rank 160 + 64 + 64)
Parent: r69a-r68a-split192-96. Type: exploit/explore.
Explicit induction without attention: for each position, a per-row sort finds the most recent earlier
occurrence of the current token; the token that followed it (the "induction candidate", causal: it is at a
position <= t) indexes a low-rank logit-prior table, folded into the head like the n-gram priors. Same
total prior rank/params as r69a (rank re-split). Confirm: < 0.926560 within the caps.
