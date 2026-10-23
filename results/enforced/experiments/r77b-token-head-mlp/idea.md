# Token-conditioned head gates + token-conditioned MLP feature selection
Parent: r76c-token-headgate
Type: exploit
Hypothesis: both token-conditioned gates helped a little (head bias r76c -0.0003, MLP gate features r76b -0.00015);
they act in different sublayers. Combined within the embedding cap: head biases (8x4) + MLP gate features for
layers 5-7 (3x32) per token = 128 per token (1.05M).
Confirm: below r76c by > 0.0003.
