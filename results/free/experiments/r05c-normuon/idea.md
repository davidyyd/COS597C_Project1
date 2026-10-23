# NorMuon: per-row second-moment normalization after Newton-Schulz

Parent: r03d-unet-skips. Explore (optimizer). NorMuon (2025): after orthogonalization, keep an EMA (beta2=0.95) of the mean squared update per output row, divide the update by its sqrt, then rescale so the update RMS is 0.2 (same as the Moonshot rule). Balances per-neuron update sizes.
