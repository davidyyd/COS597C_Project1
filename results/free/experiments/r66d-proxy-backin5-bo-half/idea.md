# proxy: backout activation from layer L/2 (not 2L/3), last 5 attention layers read it

Parent: r59a. With attention in the last 5 layers reading the cached activation, move the cache/backout point earlier: after layer L/2-1 (7) instead of 2L/3-1 (9).
