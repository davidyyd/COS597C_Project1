# MLP hidden activations normalized before the down-projection
Parent: r58b-va-gates-adam
Type: explore

Hypothesis: normalizing values (removing per-token source magnitude, so output size reflects agreement only) was
the biggest win in this search. The ReLU² MLP has the analogous magnitude problem: its hidden vector scales
quadratically with the pre-activation, so a few large activations dominate the write and its size. RMS-normalizing
the hidden pattern makes the MLP write a fixed-size mixture of its key-value "memories" weighted by relative
activation (a soft competition between hidden units).

Change: m = W_proj · norm(relu(W_fc u)^2)   (norm = gainless RMSNorm over the 2048 hidden units).

Confirm: val_loss below r58b (0.94618) by > 0.0005. Refute: equal or worse (magnitude carries useful information).

Closest work: NormFormer (LayerNorm after the MLP activation, with gains, GELU), nGPT (normalizes everything).
Here: gainless RMS normalization of a ReLU² hidden layer, motivated by the value-normalization finding, tested
in isolation.
