# MLP input gate that also sees the previous position
Parent: r63c-vag-mlp-ingate (0.94439)
Type: explore

The feature-selection gate is computed from the current input; local context may help decide which features matter:
g = 2 sigmoid(U (V u_t + V' u_{t-1})). Compare with r63c.
