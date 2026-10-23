# DDP (bucketed grad all_reduce overlapped with backward) instead of a post-backward all_reduce loop

Parent: r09a-d16w1792-canon-muonfix. Explore/exploit (efficiency). The grad all_reduce (30-48 ms/step) runs after backward; wrapping in DDP (compiled: DDPOptimizer splits the graph at buckets) overlaps it with backward compute. DDP averages instead of sums grads (Adam and Muon are scale invariant). no_sync on non-final micro-steps. Pure speed test vs r09a (2546s, 0.79473).
