# Multi-scale heads instead of query temperature
Parent: r05b-best-x0gate
Type: exploit

Multi-scale head windows (128/256/512/1024 in S layers) gave -0.0037 on baseline but hurt when added on top
of the query temperature (r03a). Hypothesis: both supply locality/sharpness control and overlap. Swap them:
best model with the query temperature removed and multi-scale windows added. Also faster (~57 ms).
Confirm: < r05b (0.95219).
