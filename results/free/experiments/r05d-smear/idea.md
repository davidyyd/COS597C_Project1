# Smear: mix the previous token's embedding into each position (gated)

Parent: r03d-unet-skips. Explore (architecture). modded-nanogpt 'smear' trick: x_t += lambda * sigmoid(W x_t[:32]) * x_{t-1} on the normalized embedding, giving every layer cheap bigram info. lambda init 0, W init 0.
