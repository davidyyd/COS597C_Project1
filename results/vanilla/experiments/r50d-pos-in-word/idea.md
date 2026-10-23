# Position-in-word embedding (tokens since the last word start)
Parent: r24c-word-embed's word-start machinery on the r25a control (bf16 residual). Type: explore.
Hypothesis: whether a BPE piece is the 1st, 2nd, 3rd... piece of a word is strong local structure (word
continuation vs. new word) that the model must infer via attention; an embedding of min(tokens since the
last word-start token, 7) gives it directly at the input. (The word-so-far mean, r24c, did not help; this
is the position signal instead of content.)
Confirm: < 0.95726 (r25a).
