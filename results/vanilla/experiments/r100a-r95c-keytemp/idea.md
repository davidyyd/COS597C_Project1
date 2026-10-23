# r95c + key-position temperature (folded into the key RoPE tables)
Parent: r95c-r94b-gatesall-ve3 (0.921402). Type: exploit.
The query-position temperature helps (~2.9e-4, r98c). Its key-side twin k_s *= 1 + b_{l,h} log(1+s)/log(T)
lets heads favour (or damp) late keys in a sharpness-dependent way, i.e. a multiplicative, content-scaled
recency effect, at no cost (folded into the key RoPE tables like the query temperature). Confirm: < 0.921402.
