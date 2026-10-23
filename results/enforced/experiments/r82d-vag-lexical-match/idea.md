# Lexical-match channel in attention: q and k carry an un-rotated token-identity code
Parent: r81c-vag-dm012-doctemp
Type: explore

Hypothesis: much of a language model's in-context gain comes from finding earlier occurrences of the current
token (copying, induction). Heads must learn this through position-entangled q/k projections of a diluted
residual. Adding the same un-rotated, normalized per-head slice of the token embedding to q and k (with learned
per-head weights) gives every head a direct, position-free "same token" match term.
Change: after RoPE + QK-norm: q_h += a_h norm(x0_h), k_h += b_h norm(x0_h), a, b init 0 (x0_h: head-sized slice
of the normalized embedding). Score gains a_h b_h cos(x0_t, x0_j) plus cross terms.
Confirm: below r81c (0.94189) by > 0.0005.
Closest work: induction heads (mechanistic), neural cache LM (Grave 2017: output-side copy); here an input-side
lexical match term inside every attention head.
