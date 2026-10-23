# Data-dependent key scale after QK-norm (key-side sharpness)
Parent: r25a-bf16-resid. Type: explore.
Query-side token temperatures did not help (r02c), but a key-side scale is different: each token decides,
per head, how strongly it attracts attention when matched (an identifier, a named entity) vs. how
diffusely (function words), while the query's static temperature stays shared.
Change: k <- exp(w_h · x_k[:32]) norm(k), w init 0. Confirm: < 0.95726.
