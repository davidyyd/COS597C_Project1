# Best + MLP hidden activations RMS-normalized before the down-projection
Parent: r64a-ingate-instead
Type: exploit

Hypothesis: r71c showed that normalizing the ReLU² hidden vector (m = W_proj norm(relu(W_fc u)^2)) gives -0.0020
on the fast base: the MLP write becomes a fixed-size mixture of its memories weighted by relative activation,
like the value normalization. It should carry to the best model.
Change: c_proj(norm(relu(h)^2)) in every layer.
Confirm: below 0.93169 by > 0.0005.
Closest work: NormFormer's post-activation LayerNorm (with gains, GELU); here gainless RMS on ReLU² hidden.
