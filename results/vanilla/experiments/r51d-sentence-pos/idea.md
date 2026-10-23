# Sentence-position embedding (log-bucket of tokens since the last sentence end)
Parent: r50d-pos-in-word machinery on the r25a control (bf16 residual). Type: explore.
Hypothesis: position within the sentence (start of sentence vs. deep inside a long one) shapes what comes
next (capitalisation, syntax, sentence length); RoPE only gives relative positions. A log2-bucketed count
of tokens since the last ". ! ? newline" (or BOS) embedding gives it directly at the input, cheaply.
Confirm: < 0.95726 (r25a).
