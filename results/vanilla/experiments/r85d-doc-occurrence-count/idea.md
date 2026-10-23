# Document occurrence count of the current token as an input feature
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: whether (and how often) x_t has already appeared in the current document is a strong cue for
copy/repetition behaviour and for topic words, but attention has to rediscover it by matching the token
against all earlier positions. Compute it exactly and causally: stable sort of (doc, token, position) per
row; the rank within each run = number of earlier occurrences in the document; bucket {0,1,2,3,4-7,8+};
add a learned embedding (6x512, zero-init) to the token embedding. Confirm: < 0.95726.
Closest: cache LMs; repetition features in n-gram LMs; our induction candidate (r75, negative) used the
successor token; this uses only the count.
