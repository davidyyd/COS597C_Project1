# Values read the token embedding in layers without value embeddings
Parent: baseline
Type: explore

Hypothesis: value embeddings (layers 1, 3, 5, 7) help because values get direct token identity. The other
layers' values only see norm(x), where token identity is diluted. Give their value projection an undiluted
copy of x0 at no parameter cost: v = W_v(norm(x) + gamma_l x0) (gamma init 0, layers 0, 2, 4, 6).
Confirm: < baseline by > 0.001.
Closest work: value embeddings / value residual (ResFormer), undiluted reads (r10c, r17b). Differs: the
shared token embedding read only by the value projection of non-VE layers.
