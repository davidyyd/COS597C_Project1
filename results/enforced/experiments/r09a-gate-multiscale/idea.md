# Low-rank self-gate + multi-scale heads
Parent: r08b-lowrank-selfgate
Type: exploit

Multi-scale head windows (128/256/512/1024 in S layers) were a big win on baseline (-0.0037) but did not
stack with the query temperature on the scalar-gate model (r03a). The richer per-dimension gate changes what
heads can do; re-test the combination (also saves ~5 ms). Confirm: < r08b (0.95027).
