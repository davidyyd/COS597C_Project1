# Position-in-word logit prior (rank 16, folded into the head)
Parent: r50d-pos-in-word machinery on the r25a control. Type: explore.
Whether the next token continues the current word depends strongly on how many pieces the word already
has; a tiny prior indexed by min(tokens since word start, 7) gives the logits this directly.
Confirm: < 0.95726 (r25a).
