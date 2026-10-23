# Value embeddings initialized at unit scale, not normalized (mechanism test)
Parent: baseline
Type: explore (ablation of the value-embedding scale finding)

Normalizing value embeddings at use gave -0.011 (r43b). Two possible causes: (a) VE tables start ~sqrt(512)x
smaller than the projected values (init), or (b) rows of rare tokens get few gradient updates while weight decay
shrinks every row each step, so their scale collapses. Initializing the tables at unit RMS (no normalization)
fixes only (a). If this recovers most of -0.011, the init scale is the cause; if not, normalization at use is
needed (sparse updates + weight decay).
