# Value normalization before adding the value embeddings
Parent: r35a-lr37-vnorm
Type: exploit

r35a normalizes v after adding the gated value embedding. Variant: normalize only the projected values
(v = s_h norm(W_v u) + g ve), so the value-embedding contribution keeps its own learned magnitude.
Confirm: < r35a (0.94107) by > 0.0003.
