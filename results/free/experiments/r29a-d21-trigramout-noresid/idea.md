# r28b + output trigram table, resid_lambdas removed (memory)

Parent: r28b-d21-bigramout-lr12e3 (best, 0.77269) + output-side trigram (r28d: -0.0007 at d16). Peak is 78GB, so free memory: drop the learnable per-layer resid_lambdas (init 1), whose gradient needs every layer's pre-mix fp32 residual saved (~0.24GB/layer at b16).
