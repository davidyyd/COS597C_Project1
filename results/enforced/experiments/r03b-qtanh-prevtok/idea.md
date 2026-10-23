# Self-gate + query temperature + previous-token embedding in keys
Parent: r02c-selfgate-qtanh
Type: exploit

Add k = W_k(norm(x) + lambda ⊙ x0_{t-1}) (all layers) to the best model; it gave -0.0009 on top of r01b.
Confirm: < r02c by > 0.0007.
