# DIAGNOSTIC: r10a with a torch.profiler table (steps 100-102) and phase timings (steps 200-299)
Parent: r10a-r09c-logn. Type: exploit (diagnostic).

Purpose: r10a adds only the log-n query scale to r09c (free alone on the baseline) yet costs +2.3 ms/step.
Comparing kernel tables with r11c shows where the time goes.
