# proxy: backout/cache after layer 10 (not 9), last 5 attention layers read it

Parent: r59a. Caching at L/2 was worse; try one layer later (index 10) with k=5 (layers 11-15).
