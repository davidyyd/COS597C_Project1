# Head self-gate also depends on document position
Parent: r87a-rope2-doc-out
Type: exploit
Hypothesis: whether a head's retrieval is useful depends on how much document context exists (e.g. heads that
retrieve from the current document are useless at its start; in late layers they then read the neighbour document).
Add c_{l,h} f_doc to each head's gate logit (zero-init). Confirm: below r87a by > 0.0003.
