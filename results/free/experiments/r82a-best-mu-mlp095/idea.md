# best (r77b): MLP matrices use a higher momentum target (warmup to 0.95 instead of 0.92)

Parent: r77b-muwarm-to092 (best, 0.76068). Momentum target per matrix type: MLP matrices x (0.95/0.92) of the scheduled momentum (others unchanged). Momentum warmup end-points 0.92-0.95 were all within noise globally; check a split.
