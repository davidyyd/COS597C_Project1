# Normalize projected values and value embeddings separately before mixing
Parent: r35b-nolr-vnorm
Type: exploit

v = s_h norm(norm(W_v u) + g norm(ve)): both sources are unit-RMS before the gated sum, so the VE gate sets their
relative weight exactly. (On r35a it measured 69.4 ms; r35b is ~1 ms cheaper.) Confirm: < r35b (0.94174).
