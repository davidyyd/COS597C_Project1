# r44c with 2-lag (instead of 4-lag) contextual value embeddings
Parent: r44c-r42b-unet. Type: exploit (cost reduction).
ctx-VE is the most valuable component (0.005 for 10 s); most of its value may come from lag 1.
Halving the lags frees time for other components. Confirm: ≈ 0.932038 and faster.
