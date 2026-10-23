# Backout: subtract a learned fraction of the 2/3-depth residual before the final norm

Parent: r16c-d16-warmup2. Explore (architecture): modded-nanogpt 'backout' - the final representation x_L - lambda * x_{2L/3} lets the model remove intermediate-layer features that are useful for computation but not for prediction. lambda (scalar) init 0.
