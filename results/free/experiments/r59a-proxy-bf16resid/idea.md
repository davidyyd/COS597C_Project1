# proxy (r49c): bf16 residual stream

Parent: r49c-d16m65-muonlr085 (0.76547, 3180s train). Efficiency: the fp32 residual stream costs ~12 full-tensor fp32 passes per layer (est. ~100ms/step of memory traffic). Keep it in bf16 (norms still compute in fp32 internally; scalars cast). Measures time saving and quality cost.
