# Low-rank GLU on the MLP hidden units
Parent: r58b-va-gates-adam
Type: explore

Instead of gating the 512 input channels (r63c, -0.0018), gate the 2048 hidden units from the input through a rank-32
map: h <- relu(W1 u)^2 * 2 sigmoid(U V u) (AdamW gate). A cheap, low-rank GLU. Compare with r63c.
Closest work: GLU / SwiGLU (full-rank gate projection, same width as the hidden). Differs: rank-32 gate, ReLU^2 kept.
