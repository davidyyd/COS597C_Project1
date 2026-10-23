# Separately normalized value parts without the final renormalization
Parent: r36b-nolr-ve-sep
Type: exploit

r36b: v = s norm(norm(W_v u) + g norm(ve)). Variant without the outer norm: v = s (norm(W_v u) + g norm(ve)), so
a strong VE gate also makes the value louder in the average (VE layers only; others v = s norm(W_v u)).
Confirm: < r36b.
