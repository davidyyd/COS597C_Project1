# Canon via F.conv1d (speed test of same math)

Parent: r07d-canon. Explore (efficiency). Same Canon math but as a depthwise F.conv1d (cuDNN) instead of 4 shifted multiply-adds; r07d cost +23% time at width 512. Compare speed (324s) and val_bpb (0.9375).
