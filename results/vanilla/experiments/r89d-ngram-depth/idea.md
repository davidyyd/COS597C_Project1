# Document repetition features with n-gram repetition depth (control)
Parent: r86d-occ-distance (on r25a-bf16-resid). Type: explore.
count + log2 distance of x_t + depth (longest n <= 4 with the n-gram ending at t seen earlier in the doc).
Confirm: < 0.954388 (r87c, three features). Closest: infini-gram / suffix-automaton match lengths used in
n-gram LMs; here an exact causal input feature for a transformer.
