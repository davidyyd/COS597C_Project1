# d16 (r30d): average weights over the last 150 steps (7 snapshots, CPU)

Parent: r30d-d16-nosmear. Explore: checkpoint averaging (LAWA-style) of the final ~4% of training (every 25 steps, kept on CPU in fp32), loaded into the model before eval. With LR decaying linearly to 0 the gain may be small; cheap to test.
