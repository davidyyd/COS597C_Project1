# MLP reads the layer's value embedding
Parent: baseline
Type: explore

Hypothesis: the MLP-reads-x0 result (r09d, -0.0022) says MLPs benefit from undiluted token identity. In the
value-embedding layers there is a richer, layer-specific token table (the VE, 512-dim). Let the MLP read it:
u = norm(x) + beta_l ve (beta init 0; only layers 1, 3, 5, 7). No new tables.
Confirm: < r09d (0.95679).
Closest work: value embeddings (attention values only), MLP-reads-x0 (r09d). Differs: VE tables shared
between the value path and the MLP input.
