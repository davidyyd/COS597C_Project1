# Attention temperature from the exact repeat count of the token
Parent: r01b-attn-selfgate
Type: explore

Hypothesis: novelty (soft cosine with the context mean) helped as a temperature signal (r31c). A sharper
signal for copying is whether this exact token occurred before in the row: rep_t = log(1 + #earlier
occurrences)/log(T), computed with a cumulative count over 256 token buckets (token id mod 256; collisions
are noise). Per-head temperature q *= 1 + tanh(c_h rep_t), c init 0.
Confirm: < r31c (0.95429).
Closest work: cache LMs / pointer networks (copy from context), novelty temperature (r31c).
