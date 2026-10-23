# proxy (r45d) with GQA: 7 KV heads for 14 query heads

Parent: r45d. Explore (efficiency): grouped-query attention halves K/V projections (attention projections 4d^2 -> 3d^2 per layer, ~6% fewer block FLOPs) and the VE tables; if quality holds, the time goes to MLP width. XSA uses V expanded to the query heads.
