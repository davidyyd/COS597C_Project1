# r48a with no per-step host syncs

Parent: r48a-d16m7-4g-emb7-s42 (best, 0.76461, 3499s total: tight). Efficiency: the loop synchronized twice per step and read the loss with .item() every step, so the GPU idled while Python launched the next step. Now: loss read / NaN check / sync only every 100 steps; training_seconds measured from step 11 to the end. Same math.
